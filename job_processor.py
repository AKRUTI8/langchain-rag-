"""
Job Processing Module
Handles job scraping, extraction, embedding generation, and Elasticsearch indexing
"""

import json
import time
from typing import List, Dict, Optional
from datetime import datetime
import hashlib

import cloudscraper
from bs4 import BeautifulSoup
from openai import OpenAI
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer


class JobProcessor:
    """Process job listings: scrape, extract info, generate embeddings"""
    
    def __init__(self, openai_api_key: str, es_url: str = "https://41590c53d1ad492d8c80599d2beaf382.us-central1.gcp.cloud.es.io/", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize with API keys and connections
        
        Args:
            openai_api_key: OpenAI API key for job info extraction
            es_url: Elasticsearch URL
            embedding_model: Sentence transformer model name (default: all-MiniLM-L6-v2)
                           Other options: 
                           - all-mpnet-base-v2 (higher quality, 768 dims)
                           - all-MiniLM-L12-v2 (balanced, 384 dims)
                           - paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
        """
        self.client = OpenAI(api_key=openai_api_key)
        cloud_url = "https://41590c53d1ad492d8c80599d2beaf382.us-central1.gcp.cloud.es.io:443"
        api_key = "b1U1SmVKc0JVR0t1YnVPbU96WUM6ZDlWakRodWY5WERPSV92c3FFQTRUdw=="
        self.es = Elasticsearch(
           cloud_url,
            api_key=api_key
        )
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        
        # Load sentence transformer model
        print(f"🤖 Loading sentence transformer model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✓ Model loaded successfully")
        
        # Get embedding dimensions from the model
        self.embedding_dimensions = self.embedding_model.get_sentence_embedding_dimension()
        print(f"📊 Embedding dimensions: {self.embedding_dimensions}")
        
    def scrape_job_page(self, url: str) -> str:
        """Scrape text content from a job posting URL"""
        try:
            response = self.scraper.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            
            return '\n'.join(lines)
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""
    
    def extract_job_info(self, job_text: str, job_url: str) -> Dict:
        """Extract structured job information using LLM"""
        prompt = f"""Extract the following information from this job posting. Return ONLY a valid JSON object:

{{
  "job_id": "unique job identifier or reference number if mentioned",
  "job_title": "the job title",
  "role": "brief description of the role (1-2 sentences)",
  "location": "job location",
  "salary_range": "salary range if mentioned (e.g. '$100K-$150K'), otherwise 'Not specified'",
  "description": "detailed job description including responsibilities and requirements",
  "link": "{job_url}"
}}

Job Posting:
{job_text[:8000]}

Return only the JSON object, no additional text."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured job information from text. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks
            for marker in ["```json", "```"]:
                result = result.replace(marker, "")
            result = result.strip()
            
            job_info = json.loads(result)
            
            # Ensure all fields exist
            defaults = {
                "job_id": f"job-{hashlib.md5(job_url.encode()).hexdigest()[:8]}",
                "job_title": "N/A",
                "role": "N/A",
                "location": "N/A",
                "salary_range": "Not specified",
                "description": "N/A",
                "link": job_url
            }
            
            for field, default in defaults.items():
                if field not in job_info:
                    job_info[field] = default
            
            job_info['processed_at'] = datetime.utcnow().isoformat()
            
            return job_info
            
        except Exception as e:
            print(f"Error extracting job info: {e}")
            return {
                "job_id": f"error-{hashlib.md5(job_url.encode()).hexdigest()[:8]}",
                "job_title": "Error",
                "role": "Error extracting information",
                "location": "N/A",
                "salary_range": "Not specified",
                "description": "Error extracting information",
                "link": job_url,
                "processed_at": datetime.utcnow().isoformat()
            }
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using sentence transformers"""
        try:
            # Truncate if too long
            max_chars = 50000
            if len(text) > max_chars:
                text = text[:max_chars]
            
            # Generate embedding using sentence transformers
            embedding = self.embedding_model.encode(
                text,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # Convert numpy array to list for JSON serialization
            return embedding.tolist()
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def process_jobs_from_file(self, input_file: str, max_jobs: Optional[int] = None) -> List[Dict]:
        """Process jobs from JSON file"""
        try:
            with open(input_file, 'r') as f:
                jobs = json.load(f)
            
            if max_jobs:
                jobs = jobs[:max_jobs]
            
            print(f"Processing {len(jobs)} jobs...")
            
            processed_jobs = []
            
            for i, job in enumerate(jobs, 1):
                job_url = job.get('absolute_url', '')
                company = job.get('company', 'Unknown')
                
                if not job_url:
                    continue
                
                print(f"\n[{i}/{len(jobs)}] Processing: {company}")
                
                # Scrape
                job_text = self.scrape_job_page(job_url)
                if not job_text:
                    continue
                
                # Extract info
                job_info = self.extract_job_info(job_text, job_url)
                job_info['company'] = company
                
                processed_jobs.append(job_info)
                print(f"  ✓ {job_info['job_title']}")
                
                time.sleep(1)  # Rate limiting
            return processed_jobs
            
        except Exception as e:
            print(f"Error processing jobs: {e}")
            return []
    
    def save_to_json(self, jobs: List[Dict], output_file: str):
        """Save jobs to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(jobs)} jobs to {output_file}")
