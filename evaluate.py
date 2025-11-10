# evaluate.py
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
# FIX 1: Change to the more powerful/standard model for better results
MODEL_NAME = 'all-mpnet-base-v2' 
LLM_MODEL = 'gemini-2.5-flash'
MAX_RECOMMENDATIONS = 10 
DURATION_MULTIPLIER = 0.5 
MIN_RECOMMENDATIONS = 5
LOG_FILENAME = 'evaluation_log.csv' 
EMBEDDINGS_FILE = 'shl_embeddings.json' 

# Global variable for all assessment data
all_assessments = []

# --- 1. Load Data ---
try:
    df_test = pd.read_excel('Gen_AI Dataset.xlsx', sheet_name='Train-Set')
    # Group the URLs by query to get the list of ground truth assessments
    ground_truth = df_test.groupby('Query')['Assessment_url'].apply(list).to_dict()
    print(f"Loaded {len(ground_truth)} ground truth queries from the 'Train-Set' sheet (Excel).")
except Exception as e:
    print(f"Fatal Error reading data (Gen_AI Dataset.xlsx): {e}")
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


# --- URL Normalization Helper (MUST MATCH app.py) ---
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

# --- 2. LLM Initialization ---
CLIENT = None
try:
    # Use the environment variable, if available
    if os.environ.get("GEMINI_API_KEY"):
        CLIENT = genai.Client()
        # No print here to avoid repeating in loop
except Exception:
    pass # Let the function handle the client being None


# --- 3. Generative AI Stage 1: Aggressive Feature Extraction (Keyword-Only Fallback) ---

def simple_keyword_fallback(query):
    """
    Guaranteed extraction of common job titles and skills (Keyword-Only Strategy), 
    with aggressive repetition to match the embedding creation logic.
    """
    keywords = set() # Use a set to avoid duplicating extracted roles
    
    # Look for job roles/skills
    roles = re.findall(r'(java|python|sql|developer|analyst|engineer|manager|sales|consultant|leadership)', query, re.IGNORECASE)
    keywords.update(r.capitalize() for r in roles) # Capitalize for consistency
    
    # Look for general behavioral/aptitude keywords
    if any(w in query.lower() for w in ['collaborat', 'teamwork', 'leadership', 'soft skills', 'personality', 'behavior']):
        keywords.add('Personality & Behavior Competencies')
    if any(w in query.lower() for w in ['cognitive', 'aptitude', 'numerical', 'verbal', 'ability', 'skills']):
        keywords.add('Ability & Aptitude')
    if 'sales' in query.lower():
        keywords.add('Sales')

    extracted_keywords_str = ' '.join(keywords)
    
    # FIX 2: Aggressively repeat keywords to match the create_embeddings.py file
    repeated_keywords = f" {extracted_keywords_str}" * 4
        
    # CRITICAL FIX: Restore the Original Query (Semantic Anchor) + Repeated Keywords
    return query + repeated_keywords

def llm_extract_features(original_query, use_llm=True):
    """Attempts LLM expansion; falls back to simple keywords if unstable or specified."""
    
    if not use_llm:
        return simple_keyword_fallback(original_query)
        
    global CLIENT
    if not CLIENT: 
        return simple_keyword_fallback(original_query)
        
    # --- LLM Expansion Prompt (Superior if it works) ---
    system_instruction = (
        "You are an intelligent Query Feature Extraction Engine for an Assessment Recommendation System. "
        "Your task is to analyze the given hiring query or job description (JD) and extract ALL "
        "critical hiring features. Your response MUST be a single, long, comprehensive paragraph that "
        "explicitly includes and repeats the key skills and assessment types. When extracting Test Types, "
        "you MUST include: **'Knowledge & Skills', 'Personality & Behavior', 'Competencies', 'Ability & Aptitude', and 'Biodata & Situational Judgement'** "
        "wherever the query implies those needs. "
        "Maximize the number of relevant SHL category keywords and technical/behavioral skills. "
        "DO NOT add any conversational text or explanation. Only output the extracted feature paragraph."
    )
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            # FIX: Use contents as a list of strings if necessary for the client version
            response = CLIENT.models.generate_content(
                model=LLM_MODEL,
                contents=[f"Original Query/JD: {original_query}"], 
                system_instruction=system_instruction 
            )
            expanded_query = response.text.strip()
            
            # If LLM succeeds and returns output, use it
            if expanded_query:
                return expanded_query
            else:
                return simple_keyword_fallback(original_query)
                
        except (APIError, Exception) as e:
            # If API fails for any reason, use the manual keyword extraction
            if attempt == max_retries - 1:
                return simple_keyword_fallback(original_query)
            time.sleep(base_delay * (2 ** attempt))

    return simple_keyword_fallback(original_query) 

# --- 4. Core Recommendation Logic (Unified) ---

def extract_duration(query):
    """
    Parses the query text to find a duration in minutes.
    """
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

# Global model instance for efficiency (using the updated MODEL_NAME)
try:
    EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print(f"Fatal Error: Could not load SentenceTransformer model '{MODEL_NAME}'. Error: {e}")
    sys.exit(1)


def get_recommendations_urls(search_query, original_query):
    """
    Generates recommendations based on semantic search of 'search_query' and 
    duration boosting based on 'original_query', enforcing uniqueness.
    Returns a list of up to MAX_RECOMMENDATIONS assessment URLs and Names.
    """
    # 1. Embed the search query
    try:
        query_embedding = EMBEDDING_MODEL.encode(search_query, convert_to_tensor=False)
        query_embedding = np.expand_dims(query_embedding, axis=0) 
    except Exception as e:
        print(f"Error embedding query: {e}", file=sys.stderr)
        return [], []

    # 2. Extract duration from original query
    query_duration = extract_duration(original_query)

    # 3. Calculate Semantic Scores (Cosine Similarity)
    semantic_scores = cosine_similarity(query_embedding, EMBEDDINGS_MATRIX)[0]
    
    final_scores = []
    
    # 4. Apply Final Duration Boost
    for i, item in enumerate(all_assessments):
        semantic_score = semantic_scores[i]

        duration_boost_factor = 0.0
        doc_duration = item.get('duration')

        if query_duration is not None and doc_duration:
            # Set tolerance as 25% of the query duration, minimum 15 minutes
            tolerance = max(15, query_duration * 0.25) 

            # Check if duration is non-numeric, matching app.py logic
            if not isinstance(doc_duration, (int, float)):
                 doc_duration = 0
                 
            if abs(doc_duration - query_duration) <= tolerance:
                duration_boost_factor = DURATION_MULTIPLIER 

        # Final score calculation: semantic_score * (1 + boost)
        final_score = semantic_score * (1 + duration_boost_factor)
        final_scores.append(final_score)

    # 5. Get the Top N indices
    final_scores = np.array(final_scores)
    # Get indices for ALL assessments, sorted from best to worst
    all_sorted_indices = np.argsort(final_scores)[::-1] 

    # 6. Retrieve URLs while enforcing UNIQUNESS 
    predicted_urls = []
    predicted_names = []
    recommended_normalized_urls = set()
    
    for index in all_sorted_indices:
        item = all_assessments[index]
        original_url = item.get('url')
        
        # Use the normalized URL for the set check
        normalized_url = normalize_url(original_url)
        
        # CRITICAL DEDUPLICATION STEP
        if normalized_url and normalized_url not in recommended_normalized_urls:
            # Append the normalized URL for recall calculation
            predicted_urls.append(normalized_url) 
            predicted_names.append(item.get('name', 'N/A'))
            recommended_normalized_urls.add(normalized_url)
        
        # Stop once we have MAX_RECOMMENDATIONS unique items
        if len(predicted_urls) >= MAX_RECOMMENDATIONS:
            break
            
    return predicted_urls, predicted_names


# --- 5. Evaluation Function (Remains mostly the same) ---

def calculate_recall_at_10(predicted_list, actual_list):
    """Calculates Recall@10 based on normalized URLs."""
    # actual_list is already normalized when created in the main loop
    normalized_actual_set = set(actual_list)
    
    relevant_found = 0
    for url in predicted_list: # predicted_list already contains normalized URLs from get_recommendations_urls
        if url in normalized_actual_set:
            relevant_found += 1
    
    total_relevant = len(normalized_actual_set)
    return relevant_found / total_relevant if total_relevant > 0 else 0.0


def run_evaluation(model_name, strategy_name, use_llm):
    
    # We use the globally initialized EMBEDDING_MODEL for consistency
    model = EMBEDDING_MODEL
        
    recall_scores = []
    log_data = [] 
    
    print(f"\n--- Running Strategy: {strategy_name} ---")

    for query, actual_urls in tqdm(ground_truth.items(), desc="Evaluating Queries"):
        
        # Determine the search query (original or LLM enhanced)
        llm_query = llm_extract_features(query, use_llm=use_llm)
        search_query = llm_query # S2 uses LLM output; S1 uses simple_keyword_fallback output
        
        predicted_urls, predicted_names = get_recommendations_urls(
            search_query=search_query, 
            original_query=query # Always use original query for duration check
        )
        
        # Normalize the ground truth URLs for accurate comparison
        normalized_actual_urls = [normalize_url(url) for url in actual_urls]
        
        recall = calculate_recall_at_10(predicted_urls, normalized_actual_urls)
        recall_scores.append(recall)

        # Print the keywords/query used for the search
        keywords_only = llm_query.replace(query, '').strip()

        if use_llm:
            print(f"\nQUERY: {query[:40]}... -> LLM QUERY (Generated): {llm_query[:50]}...")
        else:
            print(f"\nQUERY: {query[:40]}... -> KEYWORDS USED (Extracted): {keywords_only[:50]}...") # Truncate for clean output

        
        # Log the detailed results for analysis
        log_data.append({
            'Strategy': strategy_name,
            'Original Query': query,
            'LLM Enhanced Query': llm_query,
            'Recall@10': f"{recall:.2f}",
            'Ground Truth URLs': ' || '.join(normalized_actual_urls),
            'Predicted URLs': ' || '.join(predicted_urls),
            'Predicted Names': ' || '.join(predicted_names)
        })

    # Save log file (Appending to previous logs)
    df_log = pd.DataFrame(log_data)
    if os.path.exists(LOG_FILENAME):
        df_log.to_csv(LOG_FILENAME, mode='a', header=False, index=False)
    else:
        df_log.to_csv(LOG_FILENAME, index=False)

    if recall_scores:
        mean_recall_at_10 = sum(recall_scores) / len(recall_scores)
        print(f"\n[{strategy_name}] Mean Recall @ 10: {mean_recall_at_10:.4f} ({mean_recall_at_10 * 100:.2f}%)")
        print(f"Detailed log appended to {LOG_FILENAME}")
    else:
        print(f"[{strategy_name}] No scores calculated.")


# --- 6. Run Evaluation ---
if __name__ == "__main__":
    
    # Scenario 1: Keyword-Only (Guaranteed Baseline)
    run_evaluation(
        model_name=MODEL_NAME,
        strategy_name='S1_SEMANTIC_ONLY_BOOSTED',
        use_llm=False
    )

    # Scenario 2: Full LLM Attempt (Superior if it works, falls back gracefully)
    run_evaluation(
        model_name=MODEL_NAME,
        strategy_name='S2_LLM_ENRICHED_BOOSTED',
        use_llm=True
    )