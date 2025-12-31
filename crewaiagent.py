import json
import cloudscraper
from bs4 import BeautifulSoup
from openai import OpenAI
import time
from typing import List, Dict
import csv
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from datetime import datetime
import numpy as np
import PyPDF2
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large" for better quality
EMBEDDING_DIMENSIONS = 1536  # dimensions for text-embedding-3-small

# Get the ES URL, defaulting to localhost
es_url = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")

# Connect WITHOUT authentication
es = Elasticsearch(es_url)

# Initialize cloudscraper
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'mobile': False
    }
)

def create_embeddings_index(index_name: str = "job_embeddings"):
    """Create Elasticsearch index with dense_vector mapping for job embeddings."""
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
            "analysis": {
                "analyzer": {
                    "job_analyzer": {
                        "type": "standard",
                        "stopwords": "_english_"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "job_id": {
                    "type": "keyword"
                },
                "company": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    },
                    "analyzer": "job_analyzer"
                },
                "job_title": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    },
                    "analyzer": "job_analyzer"
                },
                "role": {
                    "type": "text",
                    "analyzer": "job_analyzer"
                },
                "location": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "salary_range": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "description": {
                    "type": "text",
                    "analyzer": "job_analyzer"
                },
                "link": {
                    "type": "keyword"
                },
                "description_embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIMENSIONS,
                    "index": True,
                    "similarity": "cosine"
                },
                "indexed_at": {
                    "type": "date"
                },
                "processed_at": {
                    "type": "date"
                },
                "embedding_created_at": {
                    "type": "date"
                }
            }
        }
    }
    
    try:
        if es.indices.exists(index=index_name):
            print(f"⚠ Index '{index_name}' already exists.")
            response = input(f"Do you want to delete and recreate it? (yes/no): ")
            if response.lower() in ['yes', 'y']:
                es.indices.delete(index=index_name)
                print(f"✓ Deleted existing index '{index_name}'")
                es.indices.create(index=index_name, body=mapping)
                print(f"✓ Created new Elasticsearch index with embeddings: {index_name}")
            else:
                print(f"Using existing index '{index_name}'")
        else:
            es.indices.create(index=index_name, body=mapping)
            print(f"✓ Created Elasticsearch index with embeddings: {index_name}")
            
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        raise

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI's embedding model."""
    try:
        # Truncate text if too long (max 8191 tokens for text-embedding-3-small)
        max_chars = 30000  # Conservative estimate
        if len(text) > max_chars:
            text = text[:max_chars]
        
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        
        embedding = response.data[0].embedding
        return embedding
        
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

def create_embeddings_for_jobs(jobs: List[Dict], batch_size: int = 20) -> List[Dict]:
    """Create embeddings for job descriptions in batches."""
    print("\n" + "="*60)
    print(f"Creating embeddings for {len(jobs)} jobs...")
    print("="*60)
    
    jobs_with_embeddings = []
    
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(jobs)-1)//batch_size + 1} ({len(batch)} jobs)...")
        
        for j, job in enumerate(batch, 1):
            job_copy = job.copy()
            description = job.get('description', '')
            
            if not description or description == 'N/A':
                print(f"  [{i+j}/{len(jobs)}] Skipping job without description: {job.get('job_title', 'Unknown')}")
                continue
            
            print(f"  [{i+j}/{len(jobs)}] Creating embedding for: {job.get('job_title', 'Unknown')} at {job.get('company', 'Unknown')}")
            
            # Generate embedding
            embedding = generate_embedding(description)
            
            if embedding:
                job_copy['description_embedding'] = embedding
                job_copy['embedding_created_at'] = datetime.utcnow().isoformat()
                jobs_with_embeddings.append(job_copy)
                print(f"      ✓ Embedding created ({len(embedding)} dimensions)")
            else:
                print(f"      ⚠ Failed to create embedding")
            
            # Rate limiting to avoid API limits
            time.sleep(0.5)
        
        # Longer pause between batches
        if i + batch_size < len(jobs):
            print(f"\n  Waiting 2 seconds before next batch...")
            time.sleep(2)
    
    print(f"\n✓ Created embeddings for {len(jobs_with_embeddings)}/{len(jobs)} jobs")
    return jobs_with_embeddings

def index_jobs_with_embeddings(jobs_with_embeddings: List[Dict], index_name: str = "job_embeddings"):
    """Index jobs with embeddings to Elasticsearch using bulk API."""
    if not jobs_with_embeddings:
        print("No jobs with embeddings to index.")
        return 0
    
    try:
        # Prepare bulk index actions
        actions = []
        for job in jobs_with_embeddings:
            # Add indexing timestamp
            job_copy = job.copy()
            job_copy['indexed_at'] = datetime.utcnow().isoformat()
            
            action = {
                "_index": index_name,
                "_source": job_copy
            }
            actions.append(action)
        
        # Bulk index
        success, failed = bulk(es, actions, raise_on_error=False, stats_only=False)
        
        print(f"✓ Successfully indexed {success} jobs with embeddings to '{index_name}'")
        
        if failed:
            print(f"⚠ Failed to index {len(failed)} jobs")
            for item in failed[:5]:  # Show first 5 errors
                print(f"  Error: {item}")
        
        # Refresh index to make documents immediately searchable
        es.indices.refresh(index=index_name)
        print(f"✓ Index refreshed - documents are now searchable")
        
        return success
        
    except Exception as e:
        print(f"❌ Error indexing to Elasticsearch: {e}")
        return 0

def process_and_index_embeddings(json_file: str = 'processed_jobs.json',
                                  embeddings_index: str = "job_embeddings",
                                  batch_size: int = 20,
                                  max_jobs: int = None):
    """Load processed_jobs.json, create embeddings, and index to Elasticsearch."""
    print("\n" + "="*60)
    print(f"Loading jobs from {json_file}")
    print("="*60)
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
        
        print(f"✓ Loaded {len(jobs)} jobs from {json_file}")
        
        # Limit number of jobs if specified
        if max_jobs:
            jobs = jobs[:max_jobs]
            print(f"Processing first {max_jobs} jobs...")
        
        # Create embeddings index
        create_embeddings_index(embeddings_index)
        
        # Create embeddings for jobs
        jobs_with_embeddings = create_embeddings_for_jobs(jobs, batch_size)
        
        if not jobs_with_embeddings:
            print("❌ No embeddings were created. Aborting indexing.")
            return []
        
        # Save jobs with embeddings to a new file
        embeddings_file = 'processed_jobs_with_embeddings.json'
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(jobs_with_embeddings, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved jobs with embeddings to {embeddings_file}")
        
        # Index to Elasticsearch
        indexed_count = index_jobs_with_embeddings(jobs_with_embeddings, embeddings_index)
        
        print(f"\n{'='*60}")
        print(f"✓ Total: {indexed_count} jobs with embeddings indexed successfully!")
        print(f"{'='*60}")
        
        return jobs_with_embeddings
        
    except FileNotFoundError:
        print(f"❌ Error: {json_file} not found.")
        print("Please run the job processing first to generate this file.")
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {json_file}")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def semantic_search_jobs(query: str, index_name: str = "job_embeddings", size: int = 10):
    """Perform semantic search using embeddings."""
    print(f"\n{'='*60}")
    print(f"Semantic Search: '{query}'")
    print("="*60)
    
    try:
        # Generate embedding for query
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []
        
        # Perform kNN search
        search_body = {
            "knn": {
                "field": "description_embedding",
                "query_vector": query_embedding,
                "k": size,
                "num_candidates": 100
            },
            "_source": ["job_id", "company", "job_title", "role", "location", "salary_range", "description", "link"]
        }
        
        response = es.search(index=index_name, body=search_body)
        
        hits = response['hits']['hits']
        print(f"\nFound {len(hits)} matching jobs:\n")
        
        results = []
        for i, hit in enumerate(hits, 1):
            source = hit['_source']
            score = hit['_score']
            
            print(f"{i}. {source.get('job_title', 'N/A')} at {source.get('company', 'N/A')}")
            print(f"   Job ID: {source.get('job_id', 'N/A')}")
            print(f"   Location: {source.get('location', 'N/A')}")
            print(f"   Salary: {source.get('salary_range', 'Not specified')}")
            print(f"   Similarity Score: {score:.4f}")
            print(f"   Role: {source.get('role', 'N/A')[:150]}...")
            print(f"   Link: {source.get('link', 'N/A')}")
            print("-" * 60)
            
            results.append({
                'job_id': source.get('job_id'),
                'company': source.get('company'),
                'job_title': source.get('job_title'),
                'location': source.get('location'),
                'salary_range': source.get('salary_range'),
                'role': source.get('role'),
                'link': source.get('link'),
                'similarity_score': score
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Error performing semantic search: {e}")
        return []

def keyword_search_jobs(query: str, index_name: str = "job_embeddings", size: int = 10):
    """Perform traditional keyword-based search."""
    print(f"\n{'='*60}")
    print(f"Keyword Search: '{query}'")
    print("="*60)
    
    try:
        search_body = {
            "size": size,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["job_title^3", "description^2", "role^2", "company"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["job_id", "company", "job_title", "role", "location", "salary_range", "description", "link"]
        }
        
        response = es.search(index=index_name, body=search_body)
        
        hits = response['hits']['hits']
        print(f"\nFound {len(hits)} matching jobs:\n")
        
        results = []
        for i, hit in enumerate(hits, 1):
            source = hit['_source']
            score = hit['_score']
            
            print(f"{i}. {source.get('job_title', 'N/A')} at {source.get('company', 'N/A')}")
            print(f"   Job ID: {source.get('job_id', 'N/A')}")
            print(f"   Location: {source.get('location', 'N/A')}")
            print(f"   Salary: {source.get('salary_range', 'Not specified')}")
            print(f"   Keyword Score: {score:.4f}")
            print(f"   Role: {source.get('role', 'N/A')[:150]}...")
            print(f"   Link: {source.get('link', 'N/A')}")
            print("-" * 60)
            
            results.append({
                'job_id': source.get('job_id'),
                'company': source.get('company'),
                'job_title': source.get('job_title'),
                'location': source.get('location'),
                'salary_range': source.get('salary_range'),
                'role': source.get('role'),
                'link': source.get('link'),
                'keyword_score': score
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Error performing keyword search: {e}")
        return []

def hybrid_search_jobs(query: str, index_name: str = "job_embeddings", 
                       size: int = 10, knn_weight: float = 0.7):
    """Perform hybrid search combining keyword and semantic search using RRF."""
    print(f"\n{'='*60}")
    print(f"Hybrid Search: '{query}'")
    print(f"Strategy: {int(knn_weight*100)}% Semantic + {int((1-knn_weight)*100)}% Keyword")
    print("="*60)
    
    try:
        # Generate embedding for query
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []
        
        # Step 1: Get semantic search results
        semantic_body = {
            "size": size * 3,
            "knn": {
                "field": "description_embedding",
                "query_vector": query_embedding,
                "k": size * 3,
                "num_candidates": 100
            },
            "_source": ["job_id", "company", "job_title", "role", "location", "salary_range", "description", "link"]
        }
        
        semantic_response = es.search(index=index_name, body=semantic_body)
        semantic_hits = {hit['_id']: {'doc': hit['_source'], 'score': hit['_score'], 'rank': i+1} 
                        for i, hit in enumerate(semantic_response['hits']['hits'])}
        
        # Step 2: Get keyword search results
        keyword_body = {
            "size": size * 3,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["job_title^3", "description^2", "role^2", "company"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["job_id", "company", "job_title", "role", "location", "salary_range", "description", "link"]
        }
        
        keyword_response = es.search(index=index_name, body=keyword_body)
        keyword_hits = {hit['_id']: {'doc': hit['_source'], 'score': hit['_score'], 'rank': i+1} 
                       for i, hit in enumerate(keyword_response['hits']['hits'])}
        
        # Step 3: Combine using Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant
        combined_scores = {}
        all_doc_ids = set(semantic_hits.keys()) | set(keyword_hits.keys())
        
        for doc_id in all_doc_ids:
            semantic_rank = semantic_hits.get(doc_id, {}).get('rank', size * 3 + 1)
            keyword_rank = keyword_hits.get(doc_id, {}).get('rank', size * 3 + 1)
            
            # RRF formula with weights
            rrf_score = (knn_weight / (k + semantic_rank)) + ((1 - knn_weight) / (k + keyword_rank))
            
            # Get document (prefer semantic result if available)
            doc = semantic_hits.get(doc_id, keyword_hits.get(doc_id))['doc']
            
            combined_scores[doc_id] = {
                'doc': doc,
                'rrf_score': rrf_score,
                'semantic_rank': semantic_rank if semantic_rank <= size * 3 else None,
                'keyword_rank': keyword_rank if keyword_rank <= size * 3 else None
            }
        
        # Step 4: Sort by RRF score and get top results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]['rrf_score'], reverse=True)[:size]
        
        print(f"\nFound {len(sorted_results)} matching jobs:\n")
        
        results = []
        for i, (doc_id, data) in enumerate(sorted_results, 1):
            source = data['doc']
            rrf_score = data['rrf_score']
            
            print(f"{i}. {source.get('job_title', 'N/A')} at {source.get('company', 'N/A')}")
            print(f"   Job ID: {source.get('job_id', 'N/A')}")
            print(f"   Location: {source.get('location', 'N/A')}")
            print(f"   Salary: {source.get('salary_range', 'Not specified')}")
            print(f"   RRF Score: {rrf_score:.6f}")
            
            # Show which searches found this result
            match_types = []
            if data['semantic_rank']:
                match_types.append(f"Semantic #{data['semantic_rank']}")
            if data['keyword_rank']:
                match_types.append(f"Keyword #{data['keyword_rank']}")
            print(f"   Matched by: {' & '.join(match_types)}")
            
            print(f"   Role: {source.get('role', 'N/A')[:150]}...")
            print(f"   Link: {source.get('link', 'N/A')}")
            print("-" * 60)
            
            results.append({
                'job_id': source.get('job_id'),
                'company': source.get('company'),
                'job_title': source.get('job_title'),
                'location': source.get('location'),
                'salary_range': source.get('salary_range'),
                'role': source.get('role'),
                'link': source.get('link'),
                'rrf_score': rrf_score,
                'semantic_rank': data['semantic_rank'],
                'keyword_rank': data['keyword_rank']
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Error performing hybrid search: {e}")
        import traceback
        traceback.print_exc()
        return []

def verify_embeddings_in_index(index_name: str = "job_embeddings", sample_size: int = 3):
    """Verify that embeddings are stored in Elasticsearch and show statistics."""
    print(f"\n{'='*60}")
    print(f"Verifying Embeddings in Index: '{index_name}'")
    print("="*60)
    
    try:
        # Check if index exists
        if not es.indices.exists(index=index_name):
            print(f"❌ Index '{index_name}' does not exist!")
            return
        
        # Get index mapping to verify dense_vector field
        mapping = es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]['mappings']['properties']
        
        if 'description_embedding' in properties:
            embedding_field = properties['description_embedding']
            print(f"✓ Embedding field found in mapping:")
            print(f"  - Type: {embedding_field.get('type')}")
            print(f"  - Dimensions: {embedding_field.get('dims')}")
            print(f"  - Similarity: {embedding_field.get('similarity')}")
            print(f"  - Indexed: {embedding_field.get('index')}")
        else:
            print(f"❌ No 'description_embedding' field found in mapping!")
            return
        
        # Get total document count
        count_response = es.count(index=index_name)
        total_docs = count_response['count']
        print(f"\n✓ Total documents in index: {total_docs}")
        
        # Query documents with embeddings field included
        search_body = {
            "size": sample_size,
            "query": {"match_all": {}},
            "_source": ["job_id", "company", "job_title", "salary_range", "description_embedding", "embedding_created_at"]
        }
        
        response = es.search(index=index_name, body=search_body)
        hits = response['hits']['hits']
        
        print(f"\n{'='*60}")
        print(f"Sample Documents with Embeddings (showing {len(hits)}):")
        print("="*60)
        
        docs_with_embeddings = 0
        for i, hit in enumerate(hits, 1):
            source = hit['_source']
            
            print(f"\n{i}. {source.get('job_title', 'N/A')} at {source.get('company', 'N/A')}")
            
            if 'description_embedding' in source:
                embedding = source['description_embedding']
                embedding_length = len(embedding) if embedding else 0
                
                if embedding_length > 0:
                    docs_with_embeddings += 1
                    print(f"   ✓ Embedding exists: {embedding_length} dimensions")
                    print(f"   ✓ First 5 values: {embedding[:5]}")
                    print(f"   ✓ Created at: {source.get('embedding_created_at', 'N/A')}")
                else:
                    print(f"   ⚠ Embedding field exists but is empty")
            else:
                print(f"   ❌ No embedding field in document")
        
        print(f"\n{'='*60}")
        print(f"Summary: {docs_with_embeddings}/{len(hits)} sample documents have embeddings")
        print("="*60)
        
        # Additional check: Count documents that have the embedding field
        check_body = {
            "query": {
                "exists": {
                    "field": "description_embedding"
                }
            }
        }
        
        docs_with_field = es.count(index=index_name, body=check_body)['count']
        print(f"\nTotal documents with embedding field: {docs_with_field}/{total_docs}")
        
        if docs_with_field == total_docs:
            print("✓ All documents have embeddings!")
        elif docs_with_field > 0:
            print(f"⚠ Only {docs_with_field} out of {total_docs} documents have embeddings")
        else:
            print("❌ No documents have embeddings!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_index_statistics(index_name: str = "job_embeddings"):
    """Get detailed statistics about the index."""
    print(f"\n{'='*60}")
    print(f"Index Statistics: '{index_name}'")
    print("="*60)
    
    try:
        if not es.indices.exists(index=index_name):
            print(f"❌ Index '{index_name}' does not exist!")
            return
        
        # Get index stats
        stats = es.indices.stats(index=index_name)
        index_stats = stats['indices'][index_name]
        
        print(f"\nStorage:")
        print(f"  - Total size: {index_stats['total']['store']['size_in_bytes'] / (1024*1024):.2f} MB")
        
        print(f"\nDocuments:")
        print(f"  - Count: {index_stats['total']['docs']['count']}")
        print(f"  - Deleted: {index_stats['total']['docs']['deleted']}")
        
        # Get mapping info
        mapping = es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]['mappings']['properties']
        
        print(f"\nFields:")
        for field_name, field_props in properties.items():
            field_type = field_props.get('type', 'N/A')
            if field_type == 'dense_vector':
                print(f"  - {field_name}: {field_type} ({field_props.get('dims')} dims, {field_props.get('similarity')} similarity)")
            else:
                print(f"  - {field_name}: {field_type}")
        
    except Exception as e:
        print(f"❌ Error getting index statistics: {e}")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            print(f"📄 Reading PDF: {num_pages} pages")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n"
            
            print(f"✓ Extracted {len(text)} characters from PDF")
            return text.strip()
            
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return ""

def extract_resume_info_with_llm(resume_text: str, resume_filename: str) -> Dict:
    """Use OpenAI to extract structured information from resume text."""
    prompt = f"""Extract the following information from this resume. Return ONLY a valid JSON object with these exact fields:

{{
  "resume_id": "generate a unique ID from the candidate name",
  "candidate_name": "full name of the candidate",
  "email": "email address if mentioned",
  "phone": "phone number if mentioned",
  "location": "current location or preferred location",
  "summary": "professional summary or objective (2-3 sentences)",
  "skills": ["skill1", "skill2", "skill3"],
  "experience_years": "total years of experience (number or range)",
  "education": "highest education degree and institution",
  "job_titles": ["current/recent job titles"],
  "key_achievements": "top 3-5 achievements or highlights",
  "full_text": "complete resume text for embedding"
}}

Important:
- For resume_id: create from name like "john-doe-resume"
- Extract all technical and soft skills as array
- Calculate or estimate experience_years from work history
- Include ALL resume text in full_text field for embedding

Resume:
{resume_text[:10000]}

Return only the JSON object, no additional text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured information from resumes. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        
        result = result.strip()
        
        # Parse JSON
        resume_info = json.loads(result)
        
        # Ensure all required fields are present
        required_fields = ["resume_id", "candidate_name", "email", "phone", "location", 
                          "summary", "skills", "experience_years", "education", 
                          "job_titles", "key_achievements", "full_text"]
        
        for field in required_fields:
            if field not in resume_info:
                if field == "skills" or field == "job_titles":
                    resume_info[field] = []
                elif field == "resume_id":
                    import hashlib
                    hash_str = hashlib.md5(resume_filename.encode()).hexdigest()[:8]
                    resume_info[field] = f"resume-{hash_str}"
                elif field == "full_text":
                    resume_info[field] = resume_text
                else:
                    resume_info[field] = "Not specified"
        
        # Add metadata
        resume_info['filename'] = resume_filename
        resume_info['processed_at'] = datetime.utcnow().isoformat()
        
        return resume_info
        
    except Exception as e:
        print(f"❌ Error with OpenAI API: {e}")
        import hashlib
        hash_str = hashlib.md5(resume_filename.encode()).hexdigest()[:8]
        return {
            "resume_id": f"error-{hash_str}",
            "candidate_name": "Error",
            "email": "Not specified",
            "phone": "Not specified",
            "location": "Not specified",
            "summary": "Error extracting information",
            "skills": [],
            "experience_years": "Not specified",
            "education": "Not specified",
            "job_titles": [],
            "key_achievements": "Error extracting information",
            "full_text": resume_text,
            "filename": resume_filename,
            "processed_at": datetime.utcnow().isoformat()
        }

def create_resume_embeddings_index(index_name: str = "resume_embeddings"):
    """Create Elasticsearch index for resume embeddings."""
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
            "analysis": {
                "analyzer": {
                    "resume_analyzer": {
                        "type": "standard",
                        "stopwords": "_english_"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "resume_id": {
                    "type": "keyword"
                },
                "candidate_name": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "email": {
                    "type": "keyword"
                },
                "phone": {
                    "type": "keyword"
                },
                "location": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "summary": {
                    "type": "text",
                    "analyzer": "resume_analyzer"
                },
                "skills": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    },
                    "analyzer": "resume_analyzer"
                },
                "experience_years": {
                    "type": "text"
                },
                "education": {
                    "type": "text",
                    "analyzer": "resume_analyzer"
                },
                "job_titles": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    },
                    "analyzer": "resume_analyzer"
                },
                "key_achievements": {
                    "type": "text",
                    "analyzer": "resume_analyzer"
                },
                "full_text": {
                    "type": "text",
                    "analyzer": "resume_analyzer"
                },
                "filename": {
                    "type": "keyword"
                },
                "resume_embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIMENSIONS,
                    "index": True,
                    "similarity": "cosine"
                },
                "indexed_at": {
                    "type": "date"
                },
                "processed_at": {
                    "type": "date"
                },
                "embedding_created_at": {
                    "type": "date"
                }
            }
        }
    }
    
    try:
        if es.indices.exists(index=index_name):
            print(f"⚠ Index '{index_name}' already exists.")
            response = input(f"Do you want to delete and recreate it? (yes/no): ")
            if response.lower() in ['yes', 'y']:
                es.indices.delete(index=index_name)
                print(f"✓ Deleted existing index '{index_name}'")
                es.indices.create(index=index_name, body=mapping)
                print(f"✓ Created new resume embeddings index: {index_name}")
            else:
                print(f"Using existing index '{index_name}'")
        else:
            es.indices.create(index=index_name, body=mapping)
            print(f"✓ Created resume embeddings index: {index_name}")
            
    except Exception as e:
        print(f"❌ Error creating resume index: {e}")
        raise

def process_resume_pdf(pdf_path: str, index_name: str = "resume_embeddings"):
    """Process a resume PDF: extract text, create embedding, and index to Elasticsearch."""
    print("\n" + "="*60)
    print(f"Processing Resume: {pdf_path}")
    print("="*60)
    
    try:
        # Step 1: Extract text from PDF
        resume_text = extract_text_from_pdf(pdf_path)
        
        if not resume_text:
            print("❌ Failed to extract text from PDF")
            return None
        
        # Step 2: Extract structured information using LLM
        print("\n📝 Extracting structured information with LLM...")
        resume_filename = os.path.basename(pdf_path)
        resume_info = extract_resume_info_with_llm(resume_text, resume_filename)
        
        print(f"✓ Extracted resume information:")
        print(f"  - Candidate: {resume_info.get('candidate_name')}")
        print(f"  - Email: {resume_info.get('email')}")
        print(f"  - Experience: {resume_info.get('experience_years')} years")
        print(f"  - Skills: {len(resume_info.get('skills', []))} skills")
        
        # Step 3: Generate embedding from full resume text
        print("\n🧠 Generating embedding from resume text...")
        embedding = generate_embedding(resume_info['full_text'])
        
        if not embedding:
            print("❌ Failed to generate embedding")
            return None
        
        resume_info['resume_embedding'] = embedding
        resume_info['embedding_created_at'] = datetime.utcnow().isoformat()
        print(f"✓ Embedding created ({len(embedding)} dimensions)")
        
        # Step 4: Create index if it doesn't exist
        create_resume_embeddings_index(index_name)
        
        # Step 5: Index to Elasticsearch
        print("\n💾 Indexing resume to Elasticsearch...")
        resume_info['indexed_at'] = datetime.utcnow().isoformat()
        
        try:
            response = es.index(
                index=index_name,
                id=resume_info['resume_id'],
                body=resume_info
            )
            print(f"✓ Resume indexed successfully with ID: {resume_info['resume_id']}")
            
            # Refresh index
            es.indices.refresh(index=index_name)
            
        except Exception as e:
            print(f"❌ Error indexing resume: {e}")
            return None
        
        print("\n" + "="*60)
        print("Resume Processing Complete!")
        print("="*60)
        print(f"Resume ID: {resume_info['resume_id']}")
        print(f"Index: {index_name}")
        print(f"Ready for job matching!")
        
        return resume_info
        
    except Exception as e:
        print(f"❌ Error processing resume: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_multiple_resumes(pdf_directory: str, index_name: str = "resume_embeddings"):
    """Process multiple resume PDFs from a directory."""
    print("\n" + "="*60)
    print(f"Processing Multiple Resumes from: {pdf_directory}")
    print("="*60)
    
    try:
        # Get all PDF files in directory
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_directory}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files")
        
        processed_resumes = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file}")
            print("-" * 60)
            
            resume_info = process_resume_pdf(pdf_path, index_name)
            
            if resume_info:
                processed_resumes.append(resume_info)
            
            # Rate limiting
            if i < len(pdf_files):
                time.sleep(1)
        
        print("\n" + "="*60)
        print(f"Batch Processing Complete!")
        print("="*60)
        print(f"✓ Successfully processed: {len(processed_resumes)}/{len(pdf_files)} resumes")
        
        return processed_resumes
        
    except Exception as e:
        print(f"❌ Error processing multiple resumes: {e}")
        return []

def match_resume_to_jobs(resume_id: str, resume_index: str = "resume_embeddings",
                         jobs_index: str = "job_embeddings", top_k: int = 10):
    """Match a resume to relevant jobs using embedding similarity."""
    print("\n" + "="*60)
    print(f"Matching Resume to Jobs")
    print("="*60)
    
    try:
        # Get resume from index
        resume_doc = es.get(index=resume_index, id=resume_id)
        resume_data = resume_doc['_source']
        
        print(f"Resume: {resume_data.get('candidate_name')}")
        print(f"Skills: {', '.join(resume_data.get('skills', [])[:5])}...")
        print(f"Experience: {resume_data.get('experience_years')} years")
        
        resume_embedding = resume_data.get('resume_embedding')
        
        if not resume_embedding:
            print("❌ Resume has no embedding")
            return []
        
        print(f"\n🔍 Searching for top {top_k} matching jobs...")
        
        # Search for similar jobs using kNN
        search_body = {
            "knn": {
                "field": "description_embedding",
                "query_vector": resume_embedding,
                "k": top_k,
                "num_candidates": 100
            },
            "_source": ["job_id", "company", "job_title", "role", "location", "salary_range", "description", "link"]
        }
        
        response = es.search(index=jobs_index, body=search_body)
        
        hits = response['hits']['hits']
        print(f"\n{'='*60}")
        print(f"Top {len(hits)} Job Matches:")
        print("="*60)
        
        matches = []
        for i, hit in enumerate(hits, 1):
            source = hit['_source']
            score = hit['_score']
            match_percentage = (score * 100)  # Approximate match percentage
            
            print(f"\n{i}. {source.get('job_title', 'N/A')} at {source.get('company', 'N/A')}")
            print(f"   Job ID: {source.get('job_id', 'N/A')}")
            print(f"   Location: {source.get('location', 'N/A')}")
            print(f"   Salary: {source.get('salary_range', 'Not specified')}")
            print(f"   Match Score: {score:.4f} (~{match_percentage:.1f}% match)")
            print(f"   Role: {source.get('role', 'N/A')[:150]}...")
            print(f"   Link: {source.get('link', 'N/A')}")
            print("-" * 60)
            
            matches.append({
                'job_id': source.get('job_id'),
                'company': source.get('company'),
                'job_title': source.get('job_title'),
                'location': source.get('location'),
                'salary_range': source.get('salary_range'),
                'role': source.get('role'),
                'link': source.get('link'),
                'match_score': score,
                'match_percentage': match_percentage
            })
        
        return matches
        
    except Exception as e:
        print(f"❌ Error matching resume to jobs: {e}")
        import traceback
        traceback.print_exc()
        return []

def export_resume_job_matches(resume_id: str, output_file: str = "resume_job_matches.json"):
    """Export resume and its job matches to a JSON file."""
    try:
        matches = match_resume_to_jobs(resume_id)
        
        if matches:
            # Get resume info
            resume_doc = es.get(index="resume_embeddings", id=resume_id)
            resume_data = resume_doc['_source']
            
            export_data = {
                'resume': {
                    'resume_id': resume_data.get('resume_id'),
                    'candidate_name': resume_data.get('candidate_name'),
                    'email': resume_data.get('email'),
                    'skills': resume_data.get('skills'),
                    'experience_years': resume_data.get('experience_years')
                },
                'top_matches': matches,
                'exported_at': datetime.utcnow().isoformat()
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Exported {len(matches)} job matches to {output_file}")
            return export_data
        
    except Exception as e:
        print(f"❌ Error exporting matches: {e}")
        return None

def get_index_statistics(index_name: str = "job_embeddings"):
    """Get detailed statistics about the index."""
    print(f"\n{'='*60}")
    print(f"Index Statistics: '{index_name}'")
    print("="*60)
    
    try:
        if not es.indices.exists(index=index_name):
            print(f"❌ Index '{index_name}' does not exist!")
            return
        
        # Get index stats
        stats = es.indices.stats(index=index_name)
        index_stats = stats['indices'][index_name]
        
        print(f"\nStorage:")
        print(f"  - Total size: {index_stats['total']['store']['size_in_bytes'] / (1024*1024):.2f} MB")
        
        print(f"\nDocuments:")
        print(f"  - Count: {index_stats['total']['docs']['count']}")
        print(f"  - Deleted: {index_stats['total']['docs']['deleted']}")
        
        # Get mapping info
        mapping = es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]['mappings']['properties']
        
        print(f"\nFields:")
        for field_name, field_props in properties.items():
            field_type = field_props.get('type', 'N/A')
            if field_type == 'dense_vector':
                print(f"  - {field_name}: {field_type} ({field_props.get('dims')} dims, {field_props.get('similarity')} similarity)")
            else:
                print(f"  - {field_name}: {field_type}")
        
    except Exception as e:
        print(f"❌ Error getting index statistics: {e}")

def export_sample_with_embeddings(index_name: str = "job_embeddings", 
                                  output_file: str = "sample_with_embeddings.json",
                                  sample_size: int = 2):
    """Export a sample document with embeddings to verify data structure."""
    print(f"\n{'='*60}")
    print(f"Exporting Sample Documents with Embeddings")
    print("="*60)
    
    try:
        search_body = {
            "size": sample_size,
            "query": {"match_all": {}},
            "_source": True  # Include all fields including embeddings
        }
        
        response = es.search(index=index_name, body=search_body)
        hits = response['hits']['hits']
        
        if not hits:
            print("❌ No documents found in index")
            return
        
        samples = []
        for hit in hits:
            doc = hit['_source']
            # Check if embedding exists and show metadata
            if 'description_embedding' in doc:
                embedding = doc['description_embedding']
                doc['_embedding_metadata'] = {
                    'dimensions': len(embedding),
                    'first_3_values': embedding[:3],
                    'last_3_values': embedding[-3:],
                    'has_embedding': True
                }
            else:
                doc['_embedding_metadata'] = {'has_embedding': False}
            
            samples.append(doc)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported {len(samples)} sample documents to {output_file}")
        print(f"\nYou can verify embeddings are stored by checking this file.")
        print(f"Note: Embeddings are large arrays of floats, so the file will be big.")
        
        return samples
        
    except Exception as e:
        print(f"❌ Error exporting samples: {e}")
        return None

def create_elasticsearch_index(index_name: str = "processed_jobs"):
    """Create Elasticsearch index with appropriate mappings for job data."""
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
            "analysis": {
                "analyzer": {
                    "job_analyzer": {
                        "type": "standard",
                        "stopwords": "_english_"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "job_id": {
                    "type": "keyword"
                },
                "company": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    },
                    "analyzer": "job_analyzer"
                },
                "job_title": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    },
                    "analyzer": "job_analyzer"
                },
                "role": {
                    "type": "text",
                    "analyzer": "job_analyzer"
                },
                "location": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "salary_range": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "description": {
                    "type": "text",
                    "analyzer": "job_analyzer"
                },
                "link": {
                    "type": "keyword"
                },
                "indexed_at": {
                    "type": "date"
                },
                "processed_at": {
                    "type": "date"
                }
            }
        }
    }
    
    try:
        if es.indices.exists(index=index_name):
            print(f"⚠ Index '{index_name}' already exists.")
            response = input(f"Do you want to delete and recreate it? (yes/no): ")
            if response.lower() in ['yes', 'y']:
                es.indices.delete(index=index_name)
                print(f"✓ Deleted existing index '{index_name}'")
                es.indices.create(index=index_name, body=mapping)
                print(f"✓ Created new Elasticsearch index: {index_name}")
            else:
                print(f"Using existing index '{index_name}'")
        else:
            es.indices.create(index=index_name, body=mapping)
            print(f"✓ Created Elasticsearch index: {index_name}")
            
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        raise

def index_to_elasticsearch(jobs: List[Dict], index_name: str = "processed_jobs"):
    """Index processed jobs to Elasticsearch using bulk API."""
    if not jobs:
        print("No jobs to index.")
        return 0
    
    try:
        # Prepare bulk index actions
        actions = []
        for job in jobs:
            # Add indexing timestamp
            job_copy = job.copy()
            job_copy['indexed_at'] = datetime.utcnow().isoformat()
            
            action = {
                "_index": index_name,
                "_source": job_copy
            }
            actions.append(action)
        
        # Bulk index
        success, failed = bulk(es, actions, raise_on_error=False, stats_only=False)
        
        print(f"✓ Successfully indexed {success} jobs to Elasticsearch index '{index_name}'")
        
        if failed:
            print(f"⚠ Failed to index {len(failed)} jobs")
            for item in failed:
                print(f"  Error: {item}")
        
        # Refresh index to make documents immediately searchable
        es.indices.refresh(index=index_name)
        print(f"✓ Index refreshed - documents are now searchable")
        
        return success
        
    except Exception as e:
        print(f"❌ Error indexing to Elasticsearch: {e}")
        return 0

def load_and_index_processed_jobs(json_file: str = 'processed_jobs.json', 
                                   index_name: str = "processed_jobs"):
    """Load processed_jobs.json and index to Elasticsearch."""
    print("\n" + "="*60)
    print(f"Loading jobs from {json_file}")
    print("="*60)
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
        
        print(f"✓ Loaded {len(jobs)} jobs from {json_file}")
        
        # Create index
        create_elasticsearch_index(index_name)
        
        # Index jobs
        indexed_count = index_to_elasticsearch(jobs, index_name)
        
        print(f"\n✓ Total: {indexed_count} jobs indexed successfully!")
        
        return jobs
        
    except FileNotFoundError:
        print(f"❌ Error: {json_file} not found.")
        print("Please run the job processing first to generate this file.")
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {json_file}")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []



def load_jobs(json_file_path: str) -> List[Dict]:
    """Load jobs from us_jobs.json file."""
    try:
        with open(json_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_file_path}")
        return []

def scrape_job_page(url: str) -> str:
    """Scrape text content from a job posting URL using cloudscraper."""
    try:
        response = scraper.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def extract_job_info_with_llm(job_text: str, job_url: str) -> Dict:
    """Use OpenAI to extract structured job information from scraped text."""
    prompt = f"""Extract the following information from this job posting. Return ONLY a valid JSON object with these exact fields:

{{
  "job_id": "unique job identifier or reference number if mentioned, otherwise generate from company and title",
  "job_title": "the job title",
  "role": "brief description of the role (1-2 sentences)",
  "location": "job location",
  "salary_range": "salary range if mentioned (e.g. '$100K-$150K', '₹15-25 LPA', 'Competitive'), otherwise 'Not specified'",
  "description": "detailed job description including responsibilities and requirements",
  "link": "{job_url}"
}}

Important notes:
- For job_id: Look for terms like "Job ID:", "Reference:", "Req ID:", "JR#", or similar. If not found, create one like "company-title-hash"
- For salary_range: Look for compensation, salary, pay range. Include currency and format exactly as shown. Common formats: yearly ($80K-$120K), monthly, hourly, or LPA (Lakhs Per Annum)

Job Posting:
{job_text[:8000]}

Return only the JSON object, no additional text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured job information from text. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        result = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        
        result = result.strip()
        
        # Parse JSON
        job_info = json.loads(result)
        
        # Ensure all required fields are present
        required_fields = ["job_id", "job_title", "role", "location", "salary_range", "description", "link"]
        for field in required_fields:
            if field not in job_info:
                if field == "job_id":
                    # Generate a simple job_id if not found
                    import hashlib
                    hash_str = hashlib.md5(job_url.encode()).hexdigest()[:8]
                    job_info[field] = f"job-{hash_str}"
                elif field == "salary_range":
                    job_info[field] = "Not specified"
                else:
                    job_info[field] = "N/A"
        
        # Add processing timestamp
        job_info['processed_at'] = datetime.utcnow().isoformat()
        
        return job_info
        
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        import hashlib
        hash_str = hashlib.md5(job_url.encode()).hexdigest()[:8]
        return {
            "job_id": f"error-{hash_str}",
            "job_title": "Error",
            "role": "Error extracting information",
            "location": "N/A",
            "salary_range": "Not specified",
            "description": "Error extracting information",
            "link": job_url,
            "processed_at": datetime.utcnow().isoformat()
        }

def process_jobs(input_file: str, output_json: str = 'processed_jobs.json', 
                 output_csv: str = 'processed_jobs.csv', max_jobs: int = None,
                 auto_index: bool = True, es_index_name: str = "processed_jobs"):
    """Main function to process jobs from us_jobs.json."""
    jobs = load_jobs(input_file)
    
    if not jobs:
        print("No jobs found in the input file.")
        return
    
    print(f"Found {len(jobs)} jobs to process.")
    
    if max_jobs:
        jobs = jobs[:max_jobs]
        print(f"Processing first {max_jobs} jobs...")
    
    processed_jobs = []
    
    for i, job in enumerate(jobs, 1):
        job_url = job.get('absolute_url', '')
        company = job.get('company', 'Unknown')
        
        if not job_url:
            print(f"\n[{i}/{len(jobs)}] Skipping job without URL")
            continue
        
        print(f"\n[{i}/{len(jobs)}] Processing: {company} - {job_url}")
        
        # Scrape the job page
        job_text = scrape_job_page(job_url)
        
        if not job_text:
            print("  Failed to scrape job page")
            continue
        
        print(f"  Scraped {len(job_text)} characters")
        
        # Extract info using OpenAI
        job_info = extract_job_info_with_llm(job_text, job_url)
        
        # Add company info
        job_info['company'] = company
        
        processed_jobs.append(job_info)
        print(f"  ✓ Extracted: {job_info['job_title']}")
        
        # Rate limiting
        time.sleep(1)
    
    # Save results
    print(f"\n{'='*60}")
    print(f"Total jobs processed: {len(processed_jobs)}")
    
    # Save as JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_jobs, f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved to {output_json}")
    
    # Save as CSV
    save_as_csv(processed_jobs, output_csv)
    print(f"✓ Results also saved to {output_csv}")
    
    # Auto-index to Elasticsearch
    if auto_index and processed_jobs:
        print(f"\n{'='*60}")
        print("Auto-indexing to Elasticsearch...")
        print("="*60)
        try:
            create_elasticsearch_index(es_index_name)
            indexed_count = index_to_elasticsearch(processed_jobs, es_index_name)
            print(f"✓ Auto-indexed {indexed_count} jobs to '{es_index_name}'")
        except Exception as e:
            print(f"⚠ Auto-indexing failed: {e}")
            print("You can manually index later using load_and_index_processed_jobs()")
    
    return processed_jobs

def save_as_csv(jobs: List[Dict], csv_file: str):
    """Save processed jobs to CSV file."""
    if not jobs:
        return
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['job_id', 'company', 'job_title', 'role', 'location', 'salary_range', 'description', 'link']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for job in jobs:
            writer.writerow({
                'job_id': job.get('job_id', ''),
                'company': job.get('company', ''),
                'job_title': job.get('job_title', ''),
                'role': job.get('role', ''),
                'location': job.get('location', ''),
                'salary_range': job.get('salary_range', ''),
                'description': job.get('description', ''),
                'link': job.get('link', '')
            })

if __name__ == "__main__":
    # Check if API key is loaded
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with: OPENAI_API_KEY=your-key-here")
        exit(1)
    
    print(f"✓ OpenAI API key loaded successfully")
    
    # Check Elasticsearch connection
    es_connected = False
    try:
        if es.ping():
            print(f"✓ Connected to Elasticsearch at {os.environ.get('ELASTICSEARCH_URL', 'http://localhost:9200')}")
            es_connected = True
        else:
            print("⚠ Cannot connect to Elasticsearch. Continuing without indexing...")
    except Exception as e:
        print(f"⚠ Elasticsearch connection error: {e}")
        print("Continuing without Elasticsearch indexing...")
    
    print("\n" + "="*60)
    print("JOB PROCESSING AND EMBEDDING OPTIONS")
    print("="*60)
    print("1: Process jobs from us_jobs.json and auto-index")
    print("2: Index existing processed_jobs.json to Elasticsearch")
    print("3: Create embeddings from processed_jobs.json and index")
    print("4: Semantic search with embeddings (vector only)")
    print("5: Keyword search (text only)")
    print("6: Hybrid search (keyword + semantic with RRF)")
    print("7: Verify embeddings in index")
    print("8: Get index statistics")
    print("9: Export sample documents with embeddings")
    print("\n" + "="*60)
    print("RESUME PROCESSING OPTIONS")
    print("="*60)
    print("10: Process single resume PDF and create embedding")
    print("11: Process multiple resumes from directory")
    print("12: Match resume to jobs")
    print("13: Export resume-job matches to JSON")
    print("="*60)
    
    choice = input("\nEnter your choice (1-13): ").strip()
    
    if choice == "1":
        # Process jobs from us_jobs.json with auto-indexing
        processed_jobs = process_jobs(
            input_file='us_jobs.json',
            output_json='processed_jobs.json',
            output_csv='processed_jobs.csv',
            max_jobs=10,  # Process first 10 jobs
            auto_index=es_connected,
            es_index_name='processed_jobs'
        )
        
        # Print sample results
        if processed_jobs:
            print("\n" + "="*60)
            print("Sample processed jobs:")
            print("="*60)
            for job in processed_jobs[:3]:
                print(f"\nJob ID: {job.get('job_id')}")
                print(f"Company: {job.get('company')}")
                print(f"Title: {job.get('job_title')}")
                print(f"Location: {job.get('location')}")
                print(f"Salary: {job.get('salary_range')}")
                print(f"Role: {job.get('role')}")
                print(f"Link: {job.get('link')}")
                print("-" * 60)
    
    elif choice == "2":
        # Load and index existing processed_jobs.json
        if es_connected:
            jobs = load_and_index_processed_jobs('processed_jobs.json', 'processed_jobs')
        else:
            print("❌ Cannot index without Elasticsearch connection.")
    
    elif choice == "3":
        # Create embeddings and index
        if not es_connected:
            print("❌ Cannot index without Elasticsearch connection.")
        else:
            max_jobs_input = input("Enter max number of jobs to process (or press Enter for all): ").strip()
            max_jobs = int(max_jobs_input) if max_jobs_input else None
            
            jobs_with_embeddings = process_and_index_embeddings(
                json_file='processed_jobs.json',
                embeddings_index='job_embeddings',
                batch_size=20,
                max_jobs=max_jobs
            )
            
            if jobs_with_embeddings:
                print(f"\n✓ Successfully created and indexed embeddings for {len(jobs_with_embeddings)} jobs!")
                
                # Automatically verify embeddings were stored
                print("\n" + "="*60)
                print("Automatically verifying embeddings were stored...")
                print("="*60)
                verify_embeddings_in_index('job_embeddings', sample_size=3)
    
    elif choice == "4":
        # Semantic search
        if not es_connected:
            print("❌ Cannot search without Elasticsearch connection.")
        else:
            query = input("\nEnter search query: ").strip()
            size = input("Number of results (default 10): ").strip()
            size = int(size) if size else 10
            
            results = semantic_search_jobs(query, "job_embeddings", size)
    
    elif choice == "5":
        # Keyword search
        if not es_connected:
            print("❌ Cannot search without Elasticsearch connection.")
        else:
            query = input("\nEnter search query: ").strip()
            size = input("Number of results (default 10): ").strip()
            size = int(size) if size else 10
            
            results = keyword_search_jobs(query, "job_embeddings", size)
    
    elif choice == "6":
        # Hybrid search
        if not es_connected:
            print("❌ Cannot search without Elasticsearch connection.")
        else:
            query = input("\nEnter search query: ").strip()
            size = input("Number of results (default 10): ").strip()
            size = int(size) if size else 10
            
            weight = input("Semantic weight (0.0-1.0, default 0.7): ").strip()
            weight = float(weight) if weight else 0.7
            
            results = hybrid_search_jobs(query, "job_embeddings", size, weight)
    
    elif choice == "7":
        # Verify embeddings
        if not es_connected:
            print("❌ Cannot verify without Elasticsearch connection.")
        else:
            index_to_verify = input("Enter index name (default: job_embeddings): ").strip()
            index_to_verify = index_to_verify if index_to_verify else "job_embeddings"
            
            sample_size = input("Number of samples to show (default: 3): ").strip()
            sample_size = int(sample_size) if sample_size else 3
            
            verify_embeddings_in_index(index_to_verify, sample_size)
    
    elif choice == "8":
        # Get index statistics
        if not es_connected:
            print("❌ Cannot get statistics without Elasticsearch connection.")
        else:
            index_name = input("Enter index name (default: job_embeddings): ").strip()
            index_name = index_name if index_name else "job_embeddings"
            
            get_index_statistics(index_name)
    
    elif choice == "9":
        # Export sample with embeddings
        if not es_connected:
            print("❌ Cannot export without Elasticsearch connection.")
        else:
            index_name = input("Enter index name (default: job_embeddings): ").strip()
            index_name = index_name if index_name else "job_embeddings"
            
            output_file = input("Output file (default: sample_with_embeddings.json): ").strip()
            output_file = output_file if output_file else "sample_with_embeddings.json"
            
            sample_size = input("Number of samples (default: 2): ").strip()
            sample_size = int(sample_size) if sample_size else 2
            
            export_sample_with_embeddings(index_name, output_file, sample_size)
    
    elif choice == "10":
        # Process single resume PDF
        if not es_connected:
            print("❌ Cannot process without Elasticsearch connection.")
        else:
            pdf_path = input("\nEnter path to resume PDF: ").strip()
            
            if not os.path.exists(pdf_path):
                print(f"❌ File not found: {pdf_path}")
            elif not pdf_path.lower().endswith('.pdf'):
                print("❌ File must be a PDF")
            else:
                index_name = input("Index name (default: resume_embeddings): ").strip()
                index_name = index_name if index_name else "resume_embeddings"
                
                resume_info = process_resume_pdf(pdf_path, index_name)
                
                if resume_info:
                    print("\n✓ Resume processed successfully!")
                    print(f"Resume ID: {resume_info.get('resume_id')}")
                    print(f"\nWould you like to match this resume to jobs? (yes/no)")
                    match_choice = input().strip().lower()
                    
                    if match_choice in ['yes', 'y']:
                        matches = match_resume_to_jobs(resume_info['resume_id'])
    
    elif choice == "11":
        # Process multiple resumes from directory
        if not es_connected:
            print("❌ Cannot process without Elasticsearch connection.")
        else:
            directory = input("\nEnter directory path containing resume PDFs: ").strip()
            
            if not os.path.exists(directory):
                print(f"❌ Directory not found: {directory}")
            elif not os.path.isdir(directory):
                print(f"❌ Path is not a directory: {directory}")
            else:
                index_name = input("Index name (default: resume_embeddings): ").strip()
                index_name = index_name if index_name else "resume_embeddings"
                
                resumes = process_multiple_resumes(directory, index_name)
                
                if resumes:
                    print(f"\n✓ Successfully processed {len(resumes)} resumes!")
    
    elif choice == "12":
        # Match resume to jobs
        if not es_connected:
            print("❌ Cannot match without Elasticsearch connection.")
        else:
            resume_id = input("\nEnter resume ID: ").strip()
            top_k = input("Number of top matches (default: 10): ").strip()
            top_k = int(top_k) if top_k else 10
            
            matches = match_resume_to_jobs(resume_id, top_k=top_k)
    
    elif choice == "13":
        # Export resume-job matches
        if not es_connected:
            print("❌ Cannot export without Elasticsearch connection.")
        else:
            resume_id = input("\nEnter resume ID: ").strip()
            output_file = input("Output file (default: resume_job_matches.json): ").strip()
            output_file = output_file if output_file else "resume_job_matches.json"
            
            export_resume_job_matches(resume_id, output_file)
    
    else:
        print("Invalid choice. Please run the script again and select 1-13.")