"""
Elasticsearch Manager Module
Handles index creation, data indexing, and search operations
"""

from typing import List, Dict, Optional
from datetime import datetime

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ElasticsearchManager:
    """Manage Elasticsearch indices and operations"""
    
    def __init__(self, es_url: str = "http://localhost:9200", embedding_dims: int = 1536):
        """Initialize Elasticsearch connection"""
        cloud_url = "https://41590c53d1ad492d8c80599d2beaf382.us-central1.gcp.cloud.es.io:443"
        api_key = "b1U1SmVKc0JVR0t1YnVPbU96WUM6ZDlWakRodWY5WERPSV92c3FFQTRUdw=="
        self.es = Elasticsearch(
           cloud_url,
            api_key=api_key
        )
        self.embedding_dims = embedding_dims
    
    def ping(self) -> bool:
        """Check if Elasticsearch is accessible"""
        try:
            return self.es.ping()
        except:
            return False
    
    def create_jobs_index(self, index_name: str = "job_embeddings", recreate: bool = False):
        """Create index for jobs with embeddings"""
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
                    "job_id": {"type": "keyword"},
                    "company": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                        "analyzer": "job_analyzer"
                    },
                    "job_title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                        "analyzer": "job_analyzer"
                    },
                    "role": {"type": "text", "analyzer": "job_analyzer"},
                    "location": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "salary_range": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "description": {"type": "text", "analyzer": "job_analyzer"},
                    "link": {"type": "keyword"},
                    "description_embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dims,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "indexed_at": {"type": "date"},
                    "processed_at": {"type": "date"},
                    "embedding_created_at": {"type": "date"}
                }
            }
        }
        
        if self.es.indices.exists(index=index_name):
            if recreate:
                self.es.indices.delete(index=index_name)
                print(f"✓ Deleted existing index '{index_name}'")
            else:
                print(f"⚠ Index '{index_name}' already exists")
                return
        
        self.es.indices.create(index=index_name, body=mapping)
        print(f"✓ Created index '{index_name}'")
    
    def create_resume_index(self, index_name: str = "resume_embeddings", recreate: bool = False):
        """Create index for resumes with embeddings"""
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "resume_id": {"type": "keyword"},
                    "candidate_name": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "email": {"type": "keyword"},
                    "phone": {"type": "keyword"},
                    "location": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "summary": {"type": "text"},
                    "skills": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "experience_years": {"type": "text"},
                    "education": {"type": "text"},
                    "job_titles": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "key_achievements": {"type": "text"},
                    "full_text": {"type": "text"},
                    "filename": {"type": "keyword"},
                    "resume_embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dims,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "indexed_at": {"type": "date"},
                    "processed_at": {"type": "date"},
                    "embedding_created_at": {"type": "date"}
                }
            }
        }
        
        if self.es.indices.exists(index=index_name):
            if recreate:
                self.es.indices.delete(index=index_name)
                print(f"✓ Deleted existing index '{index_name}'")
            else:
                print(f"⚠ Index '{index_name}' already exists")
                return
        
        self.es.indices.create(index=index_name, body=mapping)
        print(f"✓ Created index '{index_name}'")
    
    def bulk_index(self, documents: List[Dict], index_name: str) -> int:
        """Bulk index documents"""
        if not documents:
            return 0
        
        actions = []
        for doc in documents:
            doc_copy = doc.copy()
            doc_copy['indexed_at'] = datetime.utcnow().isoformat()
            
            actions.append({
                "_index": index_name,
                "_source": doc_copy
            })
        
        success, failed = bulk(self.es, actions, raise_on_error=False, stats_only=False)
        
        if failed:
            print(f"⚠ Failed to index {len(failed)} documents")
        
        self.es.indices.refresh(index=index_name)
        print(f"✓ Indexed {success} documents to '{index_name}'")
        
        return success
    
    def index_single(self, document: Dict, index_name: str, doc_id: Optional[str] = None):
        """Index a single document"""
        document['indexed_at'] = datetime.utcnow().isoformat()
        
        if doc_id:
            self.es.index(index=index_name, id=doc_id, body=document)
        else:
            self.es.index(index=index_name, body=document)
        
        self.es.indices.refresh(index=index_name)
    
    def semantic_search(self, query_embedding: List[float], index_name: str, 
                       embedding_field: str, size: int = 10) -> List[Dict]:
        """Perform semantic search using embeddings"""
        search_body = {
            "knn": {
                "field": embedding_field,
                "query_vector": query_embedding,
                "k": size,
                "num_candidates": 100
            }
        }
        
        response = self.es.search(index=index_name, body=search_body)
        
        results = []
        for hit in response['hits']['hits']:
            result = hit['_source']
            result['_score'] = hit['_score']
            result['_id'] = hit['_id']
            results.append(result)
        
        return results
    
    def keyword_search(self, query: str, index_name: str, 
                       fields: List[str], size: int = 10) -> List[Dict]:
        """Perform keyword-based search"""
        search_body = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
        }
        
        response = self.es.search(index=index_name, body=search_body)
        
        results = []
        for hit in response['hits']['hits']:
            result = hit['_source']
            result['_score'] = hit['_score']
            result['_id'] = hit['_id']
            results.append(result)
        
        return results
    
    def hybrid_search(self, query: str, query_embedding: List[float], 
                     index_name: str, embedding_field: str,
                     text_fields: List[str], size: int = 10,
                     semantic_weight: float = 0.7) -> List[Dict]:
        """Perform hybrid search using RRF"""
        # Get semantic results
        semantic_results = self.semantic_search(
            query_embedding, index_name, embedding_field, size * 3
        )
        semantic_dict = {r['_id']: {'data': r, 'rank': i+1} 
                        for i, r in enumerate(semantic_results)}
        
        # Get keyword results
        keyword_results = self.keyword_search(
            query, index_name, text_fields, size * 3
        )
        keyword_dict = {r['_id']: {'data': r, 'rank': i+1} 
                       for i, r in enumerate(keyword_results)}
        
        # RRF combination
        k = 60
        combined = {}
        all_ids = set(semantic_dict.keys()) | set(keyword_dict.keys())
        
        for doc_id in all_ids:
            sem_rank = semantic_dict.get(doc_id, {}).get('rank', size * 3 + 1)
            key_rank = keyword_dict.get(doc_id, {}).get('rank', size * 3 + 1)
            
            rrf_score = (semantic_weight / (k + sem_rank)) + \
                       ((1 - semantic_weight) / (k + key_rank))
            
            data = semantic_dict.get(doc_id, keyword_dict.get(doc_id))['data']
            
            combined[doc_id] = {
                'data': data,
                'rrf_score': rrf_score,
                'semantic_rank': sem_rank if sem_rank <= size * 3 else None,
                'keyword_rank': key_rank if key_rank <= size * 3 else None
            }
        
        # Sort by RRF score
        sorted_results = sorted(combined.values(), 
                               key=lambda x: x['rrf_score'], 
                               reverse=True)[:size]
        
        results = []
        for item in sorted_results:
            result = item['data']
            result['rrf_score'] = item['rrf_score']
            result['semantic_rank'] = item['semantic_rank']
            result['keyword_rank'] = item['keyword_rank']
            results.append(result)
        
        return results
    
    def get_document(self, index_name: str, doc_id: str) -> Optional[Dict]:
        """Get a single document by ID"""
        try:
            response = self.es.get(index=index_name, id=doc_id)
            return response['_source']
        except:
            return None
    
    def get_index_stats(self, index_name: str) -> Dict:
        """Get index statistics"""
        if not self.es.indices.exists(index=index_name):
            return {"error": "Index does not exist"}
        
        stats = self.es.indices.stats(index=index_name)
        count = self.es.count(index=index_name)
        
        return {
            "document_count": count['count'],
            "size_mb": stats['indices'][index_name]['total']['store']['size_in_bytes'] / (1024*1024)
        }
    
