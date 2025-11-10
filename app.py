import json
import os
import sys
import numpy as np
import re
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai.errors import APIError

# --- CONFIGURATION (MUST MATCH evaluate.py) ---
MODEL_NAME = 'all-mpnet-base-v2'
DURATION_MULTIPLIER = 0.5 
MAX_RECOMMENDATIONS = 10 # Requirement: max 10 recommendations

# --- 1. Initialize Components ---\

# Initialize Gemini Client
try:
    client = genai.Client()
except Exception:
    client = None

# Load Local Embedding Model
print(f"Loading local embedding model ({MODEL_NAME})...")
try:
    EMBEDDING_MODEL_LOCAL = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --- Basic App Setup ---\
app = Flask(__name__)
CORS(app) 

# --- Database Variables ---\
all_assessments = []
assessment_embeddings = None


def get_query_embedding(query_text):
    """Generates an embedding for the user's query using the local model."""
    try:
        return EMBEDDING_MODEL_LOCAL.encode(query_text)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None

# --- Generative AI Logic (Unstructured Expansion) ---\
def llm_rewrite_query(original_query):
    """
    Uses the Gemini API to expand and enrich the user query for better semantic matching.
    Returns only the expanded query text.
    """
    global client
    if not client:
        return original_query
        
    system_instruction = (
        "You are an intelligent Query Expansion Engine for an Assessment Recommendation System. "
        "Your task is to rewrite the given hiring query into a single, comprehensive paragraph "
        "that includes all implied skills, job roles, duration constraints (in minutes), and necessary "
        "assessment types (e.g., 'technical skills', 'behavioral competencies', 'aptitude', 'coding', 'sales'). "
        "DO NOT add any conversational text. Only output the expanded query."
    )
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Call the API
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"Original Query: {original_query}",
                config={"system_instruction": system_instruction}
            )
            
            expanded_query = response.text.strip()
            return expanded_query

        except APIError:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                return original_query
        except Exception:
            return original_query 

    return original_query

# --- Core Recommendation Logic (34.44% Winner) ---\
def get_recommendations(original_query):
    """
    Implements the winning 34.44% strategy: LLM Query Expansion + High Duration Multiplier.
    """
    # 1. LLM Query Expansion (Gets a single string)
    query_text = llm_rewrite_query(original_query)
    
    # 2. Extract Duration (from the expanded text)
    duration_match = re.search(r'(\d+)\s*(minutes?|hrs?|hours?|hour|min)', query_text, re.IGNORECASE)
    
    query_duration = None
    if duration_match:
        value = int(duration_match.group(1))
        unit = duration_match.group(2).lower()
        query_duration = value * 60 if 'hr' in unit or 'hour' in unit else value
    
    
    # 3. Semantic Search (Cosine Similarity)
    query_embedding = get_query_embedding(query_text)
    if query_embedding is None:
        return []
        
    semantic_scores = np.dot(assessment_embeddings, query_embedding)
    
    final_scores = []
    
    # 4. Apply Final Duration Boost (Multiplier)
    for i, item in enumerate(all_assessments):
        semantic_score = semantic_scores[i]
        
        duration_boost_factor = 0.0
        doc_duration = item.get('duration')
        
        if query_duration is not None and doc_duration:
            # Tolerance (e.g., +/- 15 minutes or 25% of query duration)
            tolerance = max(15, query_duration * 0.25) 
            
            if abs(doc_duration - query_duration) <= tolerance:
                duration_boost_factor = DURATION_MULTIPLIER 
                    
        # Final Score = Semantic Score * (1 + Duration Boost Factor)
        final_score = semantic_score * (1 + duration_boost_factor)
        final_scores.append(final_score)

    # 5. Get the Top N (Max 10) indices
    final_scores = np.array(final_scores)
    top_indices = np.argsort(final_scores)[-MAX_RECOMMENDATIONS:][::-1]
    
    # 6. Retrieve the full assessment data
    recommended_assessments = [all_assessments[i] for i in top_indices]
    return recommended_assessments

# =======================================================
#  ENDPOINT 1: Health Check
# =======================================================
@app.route('/health', methods=['GET'])
def health_check():
    print("Health check endpoint was hit!")
    return jsonify({"status": "healthy"}), 200

# =======================================================
#  ENDPOINT 2: Recommendation
# =======================================================
@app.route('/recommend', methods=['POST'])
def recommend_assessments():
    global all_assessments, assessment_embeddings
    
    if assessment_embeddings is None:
        return jsonify({"error": "Assessment database not loaded. Check server logs."}), 500
        
    data = request.get_json()
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({"error": "No query provided in request body."}), 400

    print(f"Received query: {user_query}")
    
    # Get the list of recommended assessment objects
    recommended_list = get_recommendations(user_query)
    
    recommendations = []
    for item in recommended_list:
        # Filter and structure the output data exactly as required by the assignment
        recommendations.append({
            "url": item.get("url"),
            "name": item.get("name"),
            "adaptive_support": item.get("adaptive_support"),
            "description": item.get("description"),
            "duration": item.get("duration"),
            "remote_support": item.get("remote_support"),
            "test_type": item.get("test_type")
        })

    return jsonify({
        "recommended_assessments": recommendations
    }), 200

# =======================================================
#  RUN THE APP
# =======================================================
if __name__ == '__main__':
    print("Starting the SHL Recommendation API...")

    # --- Load the Database on Start ---\
    print("Attempting to load assessment database...")
    try:
        with open('shl_embeddings.json', 'r') as f:
            all_assessments = json.load(f)
        
        # Ensure that every item has an embedding array
        assessment_embeddings = np.array([item['embedding'] for item in all_assessments if 'embedding' in item])
        
        # Filter all_assessments to only include items that successfully loaded an embedding
        all_assessments = [item for item in all_assessments if 'embedding' in item]
        
        print(f"Loaded {len(all_assessments)} assessments into memory.")
        print(f"Embedding matrix shape: {assessment_embeddings.shape}")
        
    except FileNotFoundError:
        print("="*50)
        print(" WARNING: 'shl_embeddings.json' not found.")
        print(" Please run 'python create_embeddings.py' first!")
        print("="*50)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading 'shl_embeddings.json': {e}")
        sys.exit(1)

    # Note: Flask runs on 127.0.0.1:5000 by default in development mode
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)