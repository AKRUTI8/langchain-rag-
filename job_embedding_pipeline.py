"""
Job Embedding Pipeline
Complete pipeline for processing jobs and creating embeddings
"""

import time
from typing import List, Dict, Optional

from job_processor import JobProcessor
from elasticsearch_manager import ElasticsearchManager


class JobEmbeddingPipeline:
    """Complete pipeline for job processing and embedding"""
    
    def __init__(self, openai_api_key: str, es_url: str = "https://41590c53d1ad492d8c80599d2beaf382.us-central1.gcp.cloud.es.io/"):
        """Initialize pipeline components"""
        self.job_processor = JobProcessor(openai_api_key, es_url)
        self.es_manager = ElasticsearchManager()
    
    def process_and_index_jobs(self, input_file: str, index_name: str = "job_embeddings",
                               max_jobs: Optional[int] = None, 
                               recreate_index: bool = False) -> int:
        """
        Complete pipeline: process jobs from file and index with embeddings
        
        Args:
            input_file: Path to JSON file with job listings
            index_name: Elasticsearch index name
            max_jobs: Maximum number of jobs to process
            recreate_index: Whether to recreate the index
            
        Returns:
            Number of jobs indexed
        """
        print("="*60)
        print("JOB EMBEDDING PIPELINE")
        print("="*60)
        
        # Step 1: Create index
        print("\n1. Creating Elasticsearch index...")
        self.es_manager.create_jobs_index(index_name, recreate=recreate_index)
        
        # Step 2: Process jobs
        print("\n2. Processing jobs from file...")
        processed_jobs = self.job_processor.process_jobs_from_file(input_file, max_jobs)
        
        if not processed_jobs:
            print("❌ No jobs were processed")
            return 0
        
        print(f"\n✓ Processed {len(processed_jobs)} jobs")
        
        # Step 3: Generate embeddings
        print("\n3. Generating embeddings...")
        jobs_with_embeddings = []
        
        for i, job in enumerate(processed_jobs, 1):
            print(f"[{i}/{len(processed_jobs)}] Embedding: {job['job_title']}")
            
            embedding = self.job_processor.generate_embedding(job['description'])
            
            if embedding:
                job['description_embedding'] = embedding
                job['embedding_created_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                jobs_with_embeddings.append(job)
                print("  ✓")
            else:
                print("  ⚠ Failed")
            
            time.sleep(0.5)  # Rate limiting
        
        print(f"\n✓ Created {len(jobs_with_embeddings)} embeddings")
        
        # Step 4: Index to Elasticsearch
        print("\n4. Indexing to Elasticsearch...")
        count = self.es_manager.bulk_index(jobs_with_embeddings, index_name)
        
        print("\n" + "="*60)
        print(f"PIPELINE COMPLETE: {count} jobs indexed")
        print("="*60)
        
        return count
    
    def search_jobs_semantic(self, query: str, index_name: str = "job_embeddings",
                            top_k: int = 10) -> List[Dict]:
        """Search jobs using semantic similarity"""
        print(f"\n🔍 Semantic Search: '{query}'")
        
        # Generate query embedding
        query_embedding = self.job_processor.generate_embedding(query)
        if not query_embedding:
            return []
        
        # Search
        results = self.es_manager.semantic_search(
            query_embedding, 
            index_name, 
            "description_embedding",
            top_k
        )
        
        print(f"✓ Found {len(results)} results")
        return results
    
    def search_jobs_keyword(self, query: str, index_name: str = "job_embeddings",
                           top_k: int = 10) -> List[Dict]:
        """Search jobs using keyword matching"""
        print(f"\n🔍 Keyword Search: '{query}'")
        
        fields = ["job_title^3", "description^2", "role^2", "company"]
        results = self.es_manager.keyword_search(query, index_name, fields, top_k)
        
        print(f"✓ Found {len(results)} results")
        return results
    
    def search_jobs_hybrid(self, query: str, index_name: str = "job_embeddings",
                          top_k: int = 10, semantic_weight: float = 0.7) -> List[Dict]:
        """Search jobs using hybrid (semantic + keyword)"""
        print(f"\n🔍 Hybrid Search: '{query}'")
        print(f"   {int(semantic_weight*100)}% Semantic + {int((1-semantic_weight)*100)}% Keyword")
        
        # Generate query embedding
        query_embedding = self.job_processor.generate_embedding(query)
        if not query_embedding:
            return []
        
        # Search
        fields = ["job_title^3", "description^2", "role^2", "company"]
        results = self.es_manager.hybrid_search(
            query,
            query_embedding,
            index_name,
            "description_embedding",
            fields,
            top_k,
            semantic_weight
        )
        
        print(f"✓ Found {len(results)} results")

        return results
    


    
