import requests
from bs4 import BeautifulSoup
import json
import time
import re

# --- Constants ---
BASE_URL = "https://www.shl.com"
CATALOG_URL_TEMPLATE = "https://www.shl.com/products/product-catalog/?start={}&type=1"

# --- Test Type Mapping ---
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

def check_tick_class_bs(element):
    """
    Helper function for BeautifulSoup to check for the green "yes" circle.
    
    --- THIS IS THE FIX for remote_support and adaptive_support ---
    The old class 'available-tick' was wrong.
    The correct element is a <span> with classes 'catalogue__circle -yes'.
    """
    # We use select_one which is good for complex CSS selectors
    if element.select_one("span.catalogue__circle.-yes"):
        return "Yes"
    else:
        return "No"

def scrape_catalog_data():
    """
    Crawls all pages of the catalog using 'requests' and 'bs4'.
    This version is no longer in test mode and will scrape all pages.
    """
    all_assessments_data = [] 
    start_index = 0
    increment = 12
    
    print(f"Initializing scrape from: {CATALOG_URL_TEMPLATE.format(0)}")
    
    while True:
        page_url = CATALOG_URL_TEMPLATE.format(start_index)
        print(f"Fetching catalog page: {page_url}")
        
        try:
            response = requests.get(page_url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error fetching {page_url}: {e}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        
        product_rows = soup.select('tr[data-entity-id]')
        
        if not product_rows:
            print("  No more assessment rows found on this page. Ending crawl.")
            break
            
        page_links_found = 0
        for row in product_rows:
            cols = row.select('td')
            if len(cols) < 4: continue 
            
            link_element = cols[0].select_one('a')
            if not link_element:
                continue

            name = link_element.text.strip()
            url = requests.compat.urljoin(BASE_URL, link_element.get('href'))
            
            # --- This function is now fixed ---
            remote_support = check_tick_class_bs(cols[1])
            adaptive_support = check_tick_class_bs(cols[2])
            
            partial_data = {
                "name": name,
                "url": url,
                "remote_support": remote_support,
                "adaptive_support": adaptive_support,
            }
            all_assessments_data.append(partial_data)
            page_links_found += 1
        
        print(f"  Found {page_links_found} assessments on this page.")
        
        start_index += increment
        time.sleep(0.2) # Be polite
            
    print(f"\nTotal partial assessment data found: {len(all_assessments_data)}")
    return all_assessments_data

def scrape_assessment_details(assessment_data):
    """
    Scrapes the missing details (description, duration, test_type)
    from the individual assessment page.
    """
    url = assessment_data.get("url")
    if not url: return None
        
    print(f"  Scraping details from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"    Error scraping {url}: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    try:
        # === DESCRIPTION SELECTOR (This was correct) ===
        description = "No description available."
        desc_h4 = soup.find('h4', string='Description')
        if desc_h4 and desc_h4.find_next_sibling('p'):
            description = desc_h4.find_next_sibling('p').get_text(separator="\n", strip=True)

        # === TEST TYPE SELECTOR (This is the fix) ===
        test_types = ["Unknown"]
        # This selector is now more specific to AVOID the tooltip legend
        # It only finds keys inside the <p> tag for "Test Type:"
        type_elements = soup.select("p.product-catalogue__small-text span.product-catalogue__key")
        if type_elements:
            test_types = [TEST_TYPE_MAP.get(el.text.strip(), el.text.strip()) for el in type_elements]

        # === DURATION SELECTOR (This was correct) ===
        duration = None
        duration_h4 = soup.find('h4', string='Assessment length')
        if duration_h4 and duration_h4.find_next_sibling('p'):
            duration_text = duration_h4.find_next_sibling('p').text
            match = re.search(r'(\d+)', duration_text) 
            if match:
                duration = int(match.group(1))
        
        # Update the dict with the new data
        assessment_data.update({
            "description": description.strip(),
            "test_type": test_types,
            "duration": duration, 
        })
        return assessment_data
        
    except Exception as e:
        print(f"    Failed to parse details for {url}: {e}")
        return None

# --- Main execution ---
if __name__ == "__main__":
    # Step 1: Get all partial data from the main catalog table
    partial_assessments = scrape_catalog_data()
    
    all_assessments = []
    if partial_assessments:
        print(f"\nNow scraping details for {len(partial_assessments)} assessments...")
        
        # Step 2: Visit each link to get the missing details
        for data in partial_assessments:
            completed_data = scrape_assessment_details(data)
            if completed_data:
                all_assessments.append(completed_data)
            time.sleep(0.1) # Be polite
            
    print(f"\nSuccessfully scraped and processed {len(all_assessments)} assessments.")
    
    # Step 3: Save to JSON
    with open('shl_assessments.json', 'w') as f:
        json.dump(all_assessments, f, indent=2)
        
    print("Data saved to 'shl_assessments.json'.")