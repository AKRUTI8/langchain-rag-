import requests
from bs4 import BeautifulSoup
import json

# Add headers to mimic a real browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get("https://fortune.com/ranking/fortune500/", headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the JSON-LD script tag that contains the company data
json_ld_script = soup.find('script', {'id': 'json-schema', 'type': 'application/ld+json'})

if json_ld_script:
    # Parse the JSON data
    data = json.loads(json_ld_script.string)
    
    # Extract the list of companies from itemListElement
    companies = data.get('itemListElement', [])
    
    # Extract company names into a list
    company_names = [company.get('item', {}).get('name') for company in companies if company.get('item', {}).get('name')]
    
    # Print the list
    print(f"Found {len(company_names)} companies")
    print(company_names[:10])  # Print first 10
    
    # Save to JSON file
    with open('fortune500_companies.json', 'w') as f:
        json.dump(company_names, f, indent=2)
    
    print(f"\n✓ Saved {len(company_names)} companies to fortune500_companies.json")
else:
    print("Could not find JSON-LD schema with company data")