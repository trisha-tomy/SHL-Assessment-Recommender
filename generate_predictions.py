import json
import os
import sys
import pandas as pd
import numpy as np
import re
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai.errors import APIError
from sklearn.metrics.pairwise import cosine_similarity 

# --- CONFIGURATION (MUST MATCH Final Configuration) ---
MODEL_NAME = 'all-mpnet-base-v2' 
LLM_MODEL = 'gemini-2.5-flash'
MAX_RECOMMENDATIONS = 10 
DURATION_MULTIPLIER = 0.5 
EMBEDDINGS_FILE = 'shl_embeddings.json' 

# Global variables for loaded data
all_assessments = []

# --- 1. Load Data ---
try:
    # Load from the 'Test-Set' sheet of the original XLSX file (Unlabeled queries)
    df_test_queries = pd.read_excel('Gen_AI Dataset.xlsx', sheet_name='Test-Set')
    unlabeled_queries = df_test_queries['Query'].tolist()
    print(f"Loaded {len(unlabeled_queries)} unlabeled queries from the 'Test-Set' sheet.")
except Exception as e:
    print(f"Fatal Error reading Test Data: {e}")
    sys.exit(1)

# Load Assessments and Embeddings
try:
    with open(EMBEDDINGS_FILE, 'r') as f:
        loaded_data = json.load(f)
    
    # Store full data (used for names and duration) and the matrix (used for similarity)
    embeddings_list = []
    
    for item in loaded_data:
        # Create a copy of the assessment dict and strip the embedding for the main list
        clean_item = {k: v for k, v in item.items() if k != 'embedding'}
        all_assessments.append(clean_item)
        embeddings_list.append(item['embedding'])
        
    EMBEDDINGS_MATRIX = np.array(embeddings_list)
    
    print(f"Loaded {len(all_assessments)} assessments and embeddings.")
except Exception as e:
    print(f"Error loading '{EMBEDDINGS_FILE}': {e}")
    sys.exit(1)


# --- URL Normalization Helper (FROM evaluate.py) ---
def normalize_url(url):
    """Normalizes the URL format for consistent comparison, matching app.py logic."""
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

# --- 2. LLM Initialization and Query Augmentation (FROM evaluate.py) ---
CLIENT = None
try:
    if os.environ.get("GEMINI_API_KEY"):
        CLIENT = genai.Client()
except Exception:
    pass 

def simple_keyword_fallback(query):
    """Fallback keyword extraction logic (matching evaluate.py logic)."""
    keywords = set() 
    roles = re.findall(r'(java|python|sql|developer|analyst|engineer|manager|sales|consultant|leadership)', query, re.IGNORECASE)
    keywords.update(r.capitalize() for r in roles)
    if any(w in query.lower() for w in ['collaborat', 'teamwork', 'leadership', 'soft skills', 'personality', 'behavior']):
        keywords.add('Personality & Behavior Competencies')
    if any(w in query.lower() for w in ['cognitive', 'aptitude', 'numerical', 'verbal', 'ability', 'skills']):
        keywords.add('Ability & Aptitude')
    if 'sales' in query.lower():
        keywords.add('Sales')

    extracted_keywords_str = ' '.join(keywords)
    # This factor must match the logic used in create_embeddings.py
    repeated_keywords = f" {extracted_keywords_str}" * 4 # Assuming 4x repetition for max performance
    return query + repeated_keywords

def llm_extract_features(original_query):
    """Attempts LLM expansion, falls back to simple keywords."""
    global CLIENT
    if not CLIENT: 
        return simple_keyword_fallback(original_query)
        
    # --- LLM Expansion Prompt (Superior if it works) ---
    system_instruction = (
        "You are an intelligent Query Feature Extraction Engine for an Assessment Recommendation System. "
        "Your task is to analyze the given hiring query or job description (JD) and extract ALL "
        "critical hiring features. Your response MUST be a single, long, comprehensive paragraph that "
        "explicitly includes and repeats the key skills and assessment types. Maximize the number of "
        "relevant SHL category keywords and technical/behavioral skills. "
        "DO NOT add any conversational text or explanation. Only output the extracted feature paragraph."
    )
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = CLIENT.models.generate_content(
                model=LLM_MODEL,
                contents=[f"Original Query/JD: {original_query}"], 
                system_instruction=system_instruction 
            )
            expanded_query = response.text.strip()
            if expanded_query:
                return expanded_query
            else:
                return simple_keyword_fallback(original_query)
        except (APIError, Exception):
            if attempt == max_retries - 1:
                return simple_keyword_fallback(original_query)
            time.sleep(base_delay * (2 ** attempt))

    return simple_keyword_fallback(original_query) 

# --- 3. Core Recommendation Logic (Unified) ---

def extract_duration(query):
    """Parses the query text to find duration in minutes (FROM evaluate.py)."""
    match_mins = re.search(r'(\d+)\s*(?:minutes|mins|min)', query, re.IGNORECASE)
    if match_mins: return int(match_mins.group(1))
    match_hours = re.search(r'(\d+)\s*hour(?:s)?', query, re.IGNORECASE)
    if match_hours: return int(match_hours.group(1)) * 60
    match_max = re.search(r'(?:max|duration|not more than)\s+[\w\s]*?(\d+)', query, re.IGNORECASE)
    if match_max:
        num = int(match_max.group(1))
        if 5 <= num <= 180: return num
    return None

try:
    EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print(f"Fatal Error: Could not load SentenceTransformer model '{MODEL_NAME}'. Error: {e}")
    sys.exit(1)


def get_recommendations_urls(original_query):
    """
    Generates recommendations based on semantic search, duration boosting, and UNIQUNESS.
    """
    # 1. LLM Query Expansion
    search_query = llm_extract_features(original_query)

    # 2. Embed the search query
    try:
        query_embedding = EMBEDDING_MODEL.encode(search_query, convert_to_tensor=False)
        query_embedding = np.expand_dims(query_embedding, axis=0) 
    except Exception as e:
        print(f"Error embedding query: {e}", file=sys.stderr)
        return []

    # 3. Extract duration from original query
    query_duration = extract_duration(original_query)

    # 4. Calculate Semantic Scores (Cosine Similarity)
    semantic_scores = cosine_similarity(query_embedding, EMBEDDINGS_MATRIX)[0]
    
    final_scores = []
    
    # 5. Apply Final Duration Boost
    for i, item in enumerate(all_assessments):
        semantic_score = semantic_scores[i]

        duration_boost_factor = 0.0
        doc_duration = item.get('duration')

        if query_duration is not None and doc_duration:
            tolerance = max(15, query_duration * 0.25) 
            if not isinstance(doc_duration, (int, float)): doc_duration = 0
            
            if abs(doc_duration - query_duration) <= tolerance:
                duration_boost_factor = DURATION_MULTIPLIER 

        final_score = semantic_score * (1 + duration_boost_factor)
        final_scores.append(final_score)

    # 6. Get Top Indices and Enforce UNIQUNESS (FROM evaluate.py)
    final_scores = np.array(final_scores)
    all_sorted_indices = np.argsort(final_scores)[::-1] 

    predicted_urls = []
    recommended_normalized_urls = set()
    
    for index in all_sorted_indices:
        item = all_assessments[index]
        original_url = item.get('url')
        
        normalized_url = normalize_url(original_url)
        
        if normalized_url and normalized_url not in recommended_normalized_urls:
            # Append the original URL (if needed for exact submission format) or normalized URL (for consistency)
            # The assignment asks for the URL as given in SHL's catalog, so we append the original URL, 
            # but we use the normalized version for the check.
            predicted_urls.append(original_url) 
            recommended_normalized_urls.add(normalized_url)
        
        # Stop once we have MAX_RECOMMENDATIONS unique items
        if len(predicted_urls) >= MAX_RECOMMENDATIONS:
            break
            
    return predicted_urls


# --- 4. Main Generation Loop ---\
if __name__ == "__main__":
    submission_data = []

    for query in tqdm(unlabeled_queries, desc="Generating Predictions"):
        predicted_urls = get_recommendations_urls(query)

        # Format for CSV submission (Appendix 3)
        for url in predicted_urls:
            submission_data.append({
                'Query': query,
                'Assessment_url': url
            })

    # Save to CSV
    df_submission = pd.DataFrame(submission_data)
    df_submission.to_csv('predictions.csv', index=False)

    print("\nSuccessfully generated 'predictions.csv' with unique results for the unlabeled test set.")