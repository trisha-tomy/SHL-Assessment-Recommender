from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import numpy as np
import re
import os
import sys
import copy 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION (MUST MATCH other files) ---
MODEL_NAME = 'all-mpnet-base-v2'
DURATION_MULTIPLIER = 0.5 
MAX_RECOMMENDATIONS = 10 
EMBEDDINGS_FILE = 'shl_embeddings.json'

# Global variables for loaded data
ASSESSMENTS_DATA = []
EMBEDDINGS_MATRIX = None

# --- URL Normalization Helper (NEWLY ADDED) ---
def normalize_url(url):
    """Normalizes the URL format for consistent comparison, matching evaluate.py logic."""
    if not isinstance(url, str):
        return ""
    # Standardize paths (replace /solutions/products/ with /products/)
    normalized_url = url.replace("/solutions/products/", "/products/")
    # Remove protocol and domain for path-only comparison
    normalized_url = normalized_url.replace("https://www.shl.com", "")
    # Remove trailing slash for absolute consistency
    if normalized_url.endswith('/'):
        normalized_url = normalized_url[:-1]
    return normalized_url


# --- 1. Load Data and Model ---
try:
    # Load the complete data once
    with open(EMBEDDINGS_FILE, 'r') as f:
        loaded_data = json.load(f)

    ASSESSMENTS_DATA = []
    embeddings_list = []
    
    # Process the loaded data to separate assessment details from embeddings
    for i, item in enumerate(loaded_data):
        if 'embedding' not in item or not isinstance(item['embedding'], list):
            print(f"Skipping item {i}: Missing or invalid 'embedding' key.")
            continue
            
        # Create a clean copy of the assessment dictionary for the application
        clean_item = {k: v for k, v in item.items() if k != 'embedding'}
        ASSESSMENTS_DATA.append(clean_item)
        
        # Extract the embedding to build the matrix
        embeddings_list.append(item['embedding'])
        
    EMBEDDINGS_MATRIX = np.array(embeddings_list)
    
    # CRITICAL CHECK: Ensure the lists are aligned
    if len(ASSESSMENTS_DATA) != len(EMBEDDINGS_MATRIX):
        raise ValueError(f"Data loading error: Mismatched lengths. Data List ({len(ASSESSMENTS_DATA)}) != Embedding Matrix ({len(EMBEDDINGS_MATRIX)})")

    print(f"Successfully loaded {len(ASSESSMENTS_DATA)} assessments and embeddings.")

except FileNotFoundError:
    print(f"Fatal Error: The file '{EMBEDDINGS_FILE}' was not found. Please run create_embeddings.py first.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Fatal Error during data loading: {e}", file=sys.stderr)
    sys.exit(1)

try:
    # Initialize the local embedding model
    EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Fatal Error: Could not load SentenceTransformer model '{MODEL_NAME}'. Error: {e}", file=sys.stderr)
    sys.exit(1)


# --- 2. Helper Functions (Duration remains unchanged) ---

def extract_duration(query):
    # ... (Duration logic remains the same)
    match_mins = re.search(r'(\d+)\s*(?:minutes|mins|min)', query, re.IGNORECASE)
    if match_mins:
        return int(match_mins.group(1))

    match_hours = re.search(r'(\d+)\s*hour(?:s)?', query, re.IGNORECASE)
    if match_hours:
        return int(match_hours.group(1)) * 60

    match_max = re.search(r'(?:max|duration|not more than)\s+[\w\s]*?(\d+)', query, re.IGNORECASE)
    if match_max:
        num = int(match_max.group(1))
        if 5 <= num <= 180: 
             return num

    return None

def get_recommendations(query):
    """
    Generates recommendations based on semantic search and duration boosting.
    """
    if not query:
        print("Debug: Query is empty, returning [].")
        return []
    
    # CRITICAL CHECK 1: Ensure matrix and data are loaded
    if EMBEDDINGS_MATRIX is None or len(ASSESSMENTS_DATA) == 0:
        print("Error: Data or embedding matrix not initialized in get_recommendations.", file=sys.stderr)
        return []

    # 1. Embed the user query
    try:
        query_embedding = EMBEDDING_MODEL.encode(query, convert_to_tensor=False)
        query_embedding = np.expand_dims(query_embedding, axis=0) 
    except Exception as e:
        print(f"Error embedding query: {e}", file=sys.stderr)
        return []

    # 2. Extract duration from query
    query_duration = extract_duration(query)
    print(f"Debug: Extracted duration from query: {query_duration} mins")

    # 3. Calculate Semantic Scores (Cosine Similarity)
    try:
        semantic_scores = cosine_similarity(query_embedding, EMBEDDINGS_MATRIX)[0]
    except Exception as e:
        print(f"Error during cosine similarity calculation: {e}", file=sys.stderr)
        return []
    
    final_scores = []
    
    # 4. Apply Final Duration Boost
    for i, item in enumerate(ASSESSMENTS_DATA): 
        semantic_score = semantic_scores[i]

        duration_boost_factor = 0.0
        doc_duration = item.get('duration') # duration key MUST exist and be a number

        if query_duration is not None and doc_duration:
            # Check to ensure doc_duration is numeric before calculation
            if not isinstance(doc_duration, (int, float)):
                 print(f"Warning: Duration for item {i} is non-numeric: {doc_duration}")
                 doc_duration = 0 # Treat as non-match
                 
            tolerance = max(15, query_duration * 0.25) 

            if abs(doc_duration - query_duration) <= tolerance:
                duration_boost_factor = DURATION_MULTIPLIER 

        final_score = semantic_score * (1 + duration_boost_factor)
        final_scores.append(final_score)

    # 5. Get the Top N indices (we sort all, then iterate for uniqueness)
    final_scores = np.array(final_scores)
    # argsort gives indices from lowest to highest score. [::-1] reverses to descending order.
    # We take all indices to ensure we find MAX_RECOMMENDATIONS unique items.
    all_sorted_indices = np.argsort(final_scores)[::-1] 

    # 6. Retrieve the full assessment objects while enforcing UNIQUNESS
    final_results = []
    recommended_normalized_urls = set() # Changed to store NORMALIZED URLs
    
    # Function to ensure all required fields are present for the frontend
    def sanitize_assessment(item, score):
        # We ensure a default empty list if 'test_type' is missing/not a list
        test_types_list = item.get('test_type')
        if not isinstance(test_types_list, list):
             test_types_list = ['N/A']
             
        # Add the score back for the frontend display
        item_copy = copy.deepcopy(item)
        item_copy['score'] = float(score)
             
        return {
            "url": item_copy.get('url', '#'), # Use the original (non-normalized) URL for the link
            "name": item_copy.get('name', 'N/A'),
            "description": item_copy.get('description', 'No description available.'),
            "test_type": test_types_list,
            "duration": item_copy.get('duration'), 
            "remote_support": item_copy.get('remote_support', 'N/A'),
            "adaptive_support": item_copy.get('adaptive_support', 'N/A'),
            "score": item_copy['score'] # Ensure score is included
        }

    for index in all_sorted_indices:
        item = ASSESSMENTS_DATA[index]
        score = final_scores[index]
        original_url = item.get('url')
        
        # CRITICAL DEDUPLICATION STEP: Use the normalized URL for the set check
        normalized_url = normalize_url(original_url)
        
        if normalized_url and normalized_url not in recommended_normalized_urls:
            final_results.append(sanitize_assessment(item, score))
            recommended_normalized_urls.add(normalized_url)
        
        # Stop once we have MAX_RECOMMENDATIONS unique items
        if len(final_results) >= MAX_RECOMMENDATIONS:
            break
            
    print(f"Debug: Successfully generated {len(final_results)} UNIQUE recommendations.")
    return final_results


# --- 3. Flask App Initialization and Endpoints ---
app = Flask(__name__)
CORS(app) 

@app.route('/')
def main_page():
    # ... (Main page serving index.html remains the same)
    try:
        with open('index.html', 'r') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return "Error: index.html not found. Please ensure all files are in the same directory.", 404

@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    data = request.get_json(silent=True)
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"error": "Query field cannot be empty."}), 400

    try:
        recommendations = get_recommendations(query)
        # If no recommendations, return an empty list
        if not recommendations:
            return jsonify([]) 
            
        return jsonify(recommendations) 
    
    except Exception as e:
        print(f"An unexpected error occurred during recommendation for query '{query}': {e}", file=sys.stderr)
        return jsonify({"error": "An internal error occurred while processing the request."}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask application on port {port}...")
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)