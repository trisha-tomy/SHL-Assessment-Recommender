# create_embeddings.py
import json
import os
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --- 0. Data Augmentation Mapping ---
KEYWORD_MAP = {
    "Competencies": "collaboration, communication, teamwork, leadership, problem-solving, decision-making, interpersonal",
    "Personality & Behavior": "sales, motivation, drive, influence, conscientiousness, leadership potential, management",
    "Knowledge & Skills": "technical, programming, java, python, sql, excel, coding, developer, expert, experienced",
    "Ability & Aptitude": "numerical, verbal, logical, reasoning, critical thinking, abstract thinking, cognitive",
    "Biodata & Situational Judgement": "real-world scenario, professional ethics, workplace behaviour, situational assessment",
}


# --- 1. Load the Local Model ---
# Standardize model for accuracy
MODEL_NAME = 'all-mpnet-base-v2' 
print(f"Loading local embedding model ({MODEL_NAME})...")
try:
    EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model. Do you have internet for the first download? {e}")
    sys.exit(1)

def get_embedding(text_block):
    """
    Generates an embedding for a block of text using the local model.
    """
    try:
        return EMBEDDING_MODEL.encode(text_block).tolist()
    except Exception as e:
        print(f"  Error embedding text block: {e}")
        return None

# --- 2. Main Execution ---
if __name__ == "__main__":
    
    input_filename = 'shl_assessments.json'
    if not os.path.exists(input_filename):
        print(f"Error: Required file '{input_filename}' not found. Please run scraper.py first.")
        sys.exit(1)
        
    print(f"\n--- Starting Embedding Creation using {MODEL_NAME} ---")
    
    try:
        with open(input_filename, 'r') as f:
            all_assessments = json.load(f)
    except Exception as e:
        print(f"Error reading JSON from {input_filename}: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(all_assessments)} raw assessments.")
        
    assessments_with_embeddings = []
    
    for item in tqdm(all_assessments, desc="Creating embeddings"):
        test_types_list = item.get('test_type', [])
        augmented_types_str = ' '.join(test_types_list)
        
        # Add detailed keywords based on the assessment's test type
        for category in test_types_list:
            if category in KEYWORD_MAP:
                augmented_types_str += f" {KEYWORD_MAP[category]}" 

        # --- RE-CREATE THE "DATABASE STYLE" TEXT FORMAT (Reduced REPETITION) ---
        duration_text = "N/A"
        if item.get('duration'):
            duration_text = f"{item.get('duration')} minutes"

        name = item.get('name', '')
        
        # FIX: ONLY repeat the Augmented Keywords ONCE. This is the crucial change
        # to ensure the description and name aren't overwhelmed by repetitive keywords.
        repeated_keywords = f" {augmented_types_str}"
        
        text_to_embed = f"""
Assessment Name: {name}
Description: {item.get('description', '')}
Test Types: {augmented_types_str}{repeated_keywords} 
Duration: {duration_text}
Remote Support: {item.get('remote_support', 'N/A')}
Adaptive Support: {item.get('adaptive_support', 'N/A')}
"""
        # --- END OF CRITICAL FIX ---
        
        embedding = get_embedding(text_to_embed)
        
        if embedding is not None:
            # We deep copy the item before appending the large embedding list
            item_copy = item.copy() 
            item_copy['embedding'] = embedding
            assessments_with_embeddings.append(item_copy)
    
    output_filename = 'shl_embeddings.json'
    with open(output_filename, 'w') as f:
        json.dump(assessments_with_embeddings, f, indent=2)
        
    print(f"\nSuccessfully created embeddings for {len(assessments_with_embeddings)} assessments.")
    print(f"Output saved to '{output_filename}'.")