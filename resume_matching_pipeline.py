"""
Resume Matching Pipeline
Complete pipeline for processing resumes and matching to jobs
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime

from resume_processor import ResumeProcessor
from elasticsearch_manager import ElasticsearchManager


class ResumeMatchingPipeline:
    """Complete pipeline for resume processing and job matching"""
    
    def __init__(self, openai_api_key: str, es_url: str = "http://localhost:9200"):
        """Initialize pipeline components"""
        self.resume_processor = ResumeProcessor(openai_api_key)
        self.es_manager = ElasticsearchManager(es_url)
    
    def process_and_index_resume(self, pdf_path: str, 
                                 index_name: str = "resume_embeddings") -> Optional[str]:
        """
        Process a resume PDF and index to Elasticsearch
        
        Args:
            pdf_path: Path to resume PDF
            index_name: Elasticsearch index name
            
        Returns:
            Resume ID if successful, None otherwise
        """
        print("="*60)
        print("RESUME PROCESSING PIPELINE")
        print("="*60)
        
        # Ensure index exists
        self.es_manager.create_resume_index(index_name, recreate=False)
        
        # Process resume
        resume_info = self.resume_processor.process_resume_pdf(pdf_path)
        
        if not resume_info:
            print("❌ Failed to process resume")
            return None
        
        # Index to Elasticsearch
        print("\n💾 Indexing to Elasticsearch...")
        self.es_manager.index_single(
            resume_info, 
            index_name, 
            doc_id=resume_info['resume_id']
        )
        
        print(f"✓ Resume indexed: {resume_info['resume_id']}")
        print("="*60)
        
        return resume_info['resume_id']
    
    def process_multiple_resumes(self, directory: str,
                                 index_name: str = "resume_embeddings") -> List[str]:
        """
        Process multiple resume PDFs from a directory
        
        Args:
            directory: Directory containing PDF files
            index_name: Elasticsearch index name
            
        Returns:
            List of resume IDs
        """
        print("="*60)
        print("BATCH RESUME PROCESSING")
        print("="*60)
        
        # Ensure index exists
        self.es_manager.create_resume_index(index_name, recreate=False)
        
        # Get PDF files
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"❌ No PDF files found in {directory}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files\n")
        
        resume_ids = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(directory, pdf_file)
            print(f"[{i}/{len(pdf_files)}] {pdf_file}")
            print("-" * 60)
            
            # Process
            resume_info = self.resume_processor.process_resume_pdf(pdf_path)
            
            if resume_info:
                # Index
                self.es_manager.index_single(
                    resume_info,
                    index_name,
                    doc_id=resume_info['resume_id']
                )
                resume_ids.append(resume_info['resume_id'])
                print(f"✓ Indexed: {resume_info['resume_id']}\n")
            else:
                print("⚠ Failed to process\n")
        
        print("="*60)
        print(f"BATCH COMPLETE: {len(resume_ids)}/{len(pdf_files)} resumes indexed")
        print("="*60)
        
        return resume_ids
    
    def match_resume_to_jobs(self, resume_id: str,
                            resume_index: str = "resume_embeddings",
                            jobs_index: str = "job_embeddings",
                            top_k: int = 10) -> List[Dict]:
        """
        Match a resume to relevant jobs using embedding similarity
        
        Args:
            resume_id: Resume ID to match
            resume_index: Resume index name
            jobs_index: Jobs index name
            top_k: Number of top matches to return
            
        Returns:
            List of matching jobs with scores
        """
        print("="*60)
        print("RESUME-TO-JOBS MATCHING")
        print("="*60)
        
        # Get resume
        resume = self.es_manager.get_document(resume_index, resume_id)
        
        if not resume:
            print(f"❌ Resume '{resume_id}' not found")
            return []
        
        print(f"\nResume: {resume['candidate_name']}")
        print(f"Skills: {', '.join(resume.get('skills', [])[:5])}...")
        print(f"Experience: {resume.get('experience_years')}")
        
        resume_embedding = resume.get('resume_embedding')
        
        if not resume_embedding:
            print("❌ Resume has no embedding")
            return []
        
        print(f"\n🔍 Searching for top {top_k} matching jobs...")
        
        # Search
        results = self.es_manager.semantic_search(
            resume_embedding,
            jobs_index,
            "description_embedding",
            top_k
        )
        
        # Add match percentage
        for result in results:
            result['match_percentage'] = result['_score'] * 100
        
        print(f"✓ Found {len(results)} matches")
        print("="*60)
        
        return results
    
    def export_matches(self, resume_id: str, matches: List[Dict],
                      output_file: str = "resume_job_matches.json"):
        """Export resume matches to JSON file"""
        resume = self.es_manager.get_document("resume_embeddings", resume_id)
        
        export_data = {
            'resume': {
                'resume_id': resume['resume_id'],
                'candidate_name': resume['candidate_name'],
                'email': resume['email'],
                'skills': resume['skills'],
                'experience_years': resume['experience_years']
            },
            'top_matches': [{
                'job_id': m['job_id'],
                'company': m['company'],
                'job_title': m['job_title'],
                'location': m['location'],
                'salary_range': m['salary_range'],
                'match_score': m['_score'],
                'match_percentage': m['match_percentage'],
                'link': m['link']
            } for m in matches],
            'exported_at': datetime.utcnow().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        

        print(f"✓ Exported {len(matches)} matches to {output_file}")
