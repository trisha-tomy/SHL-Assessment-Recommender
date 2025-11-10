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

# --- Data Augmentation Mapping ---\
KEYWORD_MAP = {
    "Competencies": "collaboration, communication, teamwork, leadership, problem-solving, decision-making, interpersonal",
    "Personality & Behavior": "sales, motivation, drive, influence, conscientiousness, leadership potential, management",
    "Knowledge & Skills": "technical, programming, java, python, sql, excel, coding, developer, expert, experienced",
    "Ability & Aptitude": "numerical, verbal, logical, reasoning, critical thinking, abstract thinking, cognitive",
    "Biodata & Situational Judgement": "real-world scenario, professional ethics, workplace behaviour, situational assessment",
}

# --- BOOSTING CONFIGURATION ---
DURATION_MULTIPLIER = 0.5 
# Test Type Boosts are very small and CUMULATIVE (Additive)
TYPE_ADDITIVE_WEIGHTS = {
    "Knowledge & Skills": 0.03,  # Gentle boost for technical skills
    "Competencies": 0.02,
    "Ability & Aptitude": 0.01,
}
# Only include types that have a non-zero boost
BOOSTED_TYPES = list(TYPE_ADDITIVE_WEIGHTS.keys())


# --- 1. Load Data ---\
# Load ground truth data
try:
    # Load from the 'Train-Set' sheet of the original XLSX file
    df_test = pd.read_excel('Gen_AI Dataset.xlsx', sheet_name='Train-Set')
    # Group by query to get all relevant URLs for each query
    ground_truth = df_test.groupby('Query')['Assessment_url'].apply(list).to_dict()
    print(f"Loaded {len(ground_truth)} ground truth queries from the 'Train-Set' sheet (Excel).")
except Exception as e:
    print(f"Fatal Error reading data: {e}")
    sys.exit(1)


# --- URL Normalization Helper ---\
def normalize_url(url):
    """
    Standardizes the URL for comparison by removing the domain, focusing on the path.
    """
    if not isinstance(url, str):
        return ""
    
    normalized_url = url.replace("/solutions/products/", "/products/")
    normalized_url = normalized_url.replace("https://www.shl.com", "")
    
    return normalized_url

# --- 2. LLM Query Rewriting/Expansion ---\
# Initialize Gemini Client
try:
    client = genai.Client()
except Exception:
    client = None

def llm_rewrite_query(original_query):
    # (LLM function remains the same for all strategies)
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
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"Original Query: {original_query}",
                config={"system_instruction": system_instruction}
            )
            return response.text.strip()
        except APIError:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                return original_query
        except Exception:
            return original_query 
    return original_query


# --- 3. Recommendation Core Logic ---\

def get_top_10_recommendations(original_query, embedding_model, all_assessments, assessment_embeddings, boost_strategy):
    """
    Core function to retrieve and score recommendations based on a chosen strategy.
    """
    # 1. LLM Query Expansion
    expanded_query = llm_rewrite_query(original_query)
    query_text = expanded_query
    
    # 2. Extract Duration (from the LLM-expanded query)
    duration_match = re.search(r'(\d+)\s*(minutes?|hrs?|hours?|hour|min)', query_text, re.IGNORECASE)
    query_duration = None
    if duration_match:
        value = int(duration_match.group(1))
        unit = duration_match.group(2).lower()
        query_duration = value * 60 if 'hr' in unit or 'hour' in unit else value
    
    
    # --- A. Semantic Search (Cosine Similarity) ---
    query_embedding = embedding_model.encode(query_text)
    semantic_scores = np.dot(assessment_embeddings, query_embedding)
    
    final_scores = []
    
    # --- B. Apply Final Boosts ---
    for i, item in enumerate(all_assessments):
        semantic_score = semantic_scores[i]
        
        # --- Duration Boost (Multiplier) ---
        duration_multiplier = 0.0
        doc_duration = item.get('duration')
        
        if query_duration and doc_duration:
            tolerance = max(15, query_duration * 0.25) 
            if abs(doc_duration - query_duration) <= tolerance:
                duration_multiplier = DURATION_MULTIPLIER 
        
        total_boost = duration_multiplier
        final_score = semantic_score * (1 + total_boost)
        
        # --- ADDITIVE Boost for Enhanced Boosting Strategy ---
        if boost_strategy == 'ENHANCED_BOOST':
            additive_boost = 0.0
            
            # Check for Test Type matches (Additive Boost)
            for doc_type in item.get('test_type', []):
                if doc_type in BOOSTED_TYPES:
                    # Check if the type/keywords are mentioned in the LLM's query
                    keywords = [doc_type]
                    if doc_type in KEYWORD_MAP:
                        keywords.extend(KEYWORD_MAP[doc_type].split(', '))
                    
                    if any(k.strip().lower() in query_text.lower() for k in keywords):
                        additive_boost += TYPE_ADDITIVE_WEIGHTS[doc_type]
                        
            # Use Additive Boost on the final score for the ENHANCED strategy
            final_score += additive_boost


        final_scores.append(final_score)

    # Get the top 10 indices
    final_scores = np.array(final_scores)
    top_indices = np.argsort(final_scores)[-10:][::-1]
    
    # Retrieve the URLs based on the top indices
    assessment_urls = [item['url'] for item in all_assessments]
    predicted_urls = [assessment_urls[i] for i in top_indices]
    return predicted_urls


def calculate_recall_at_10(predicted_list, actual_list):
    """
    Calculates Recall@10 for a single query using robust URL normalization.
    """
    # Normalize ALL URLs before creating the sets for comparison
    normalized_actual_set = set(normalize_url(url) for url in actual_list)
    normalized_predicted_list = [normalize_url(url) for url in predicted_list]
    
    relevant_found = 0
    for url in normalized_predicted_list:
        if url in normalized_actual_set:
            relevant_found += 1
    
    total_relevant = len(normalized_actual_set)
    
    if total_relevant == 0:
        return 0.0
        
    return relevant_found / total_relevant


def run_evaluation(model_name, boost_strategy, log_name):
    """Loads model/data and runs the full evaluation loop for one strategy."""
    print(f"\n--- Running Strategy: {log_name} ---")
    
    # --- Load Local Model (Specific for this run) ---
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Skipping {log_name}: Could not load model {model_name}. Error: {e}")
        return
    
    # --- Load Embeddings (Assumes embeddings file exists for the chosen model) ---
    # NOTE: Since the embeddings file is generic ('shl_embeddings.json'), 
    # we assume 'create_embeddings.py' was last run with the correct model (MPNet in this case).
    try:
        with open('shl_embeddings.json', 'r') as f:
            assessments = json.load(f)
        
        embeddings = np.array([item['embedding'] for item in assessments])
    except FileNotFoundError:
        print("Skipping: 'shl_embeddings.json' not found. Please run 'create_embeddings.py'.")
        return
    
    recall_scores = []
    
    for query, actual_urls in tqdm(ground_truth.items(), desc="Evaluating Queries"):
        
        predicted_urls = get_top_10_recommendations(
            query, 
            model, 
            assessments, 
            embeddings, 
            boost_strategy
        )
        recall = calculate_recall_at_10(predicted_urls, actual_urls)
        recall_scores.append(recall)

    if recall_scores:
        mean_recall_at_10 = sum(recall_scores) / len(recall_scores)
        print(f"\n[{log_name}] Mean Recall @ 10: {mean_recall_at_10:.4f} ({mean_recall_at_10 * 100:.2f}%)")
    else:
        print(f"[{log_name}] No scores calculated.")


# --- 4. Run Multi-Strategy Evaluation ---\
if __name__ == "__main__":
    
    # --- Strategy 1: The Baseline Winner (MPNet + High Multiplier) ---
    # This should reproduce your ~30.44% score, confirming the Name Weighting was the issue.
    run_evaluation(
        model_name='all-mpnet-base-v2',
        boost_strategy='BASELINE',
        log_name='BASELINE_MPNET_PURE_SEMANTIC'
    )
    
    # --- Strategy 2: Enhanced Boosting (MPNet + High Multiplier + Subtle Additive Type Boost) ---
    # We test if the subtle ADDITIVE boost can push the baseline higher.
    # run_evaluation(
    #     model_name='all-mpnet-base-v2',
    #     boost_strategy='ENHANCED_BOOST',
    #     log_name='ENHANCED_MPNET_ADDITIVE_TYPE'
    # )

    # print("\n--- Summary: Re-run create_embeddings.py with BGE-Small to test Hypothesis 3! ---")