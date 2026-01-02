

import os
import json
from typing import Dict, Optional
from datetime import datetime
import hashlib
import re

import PyPDF2
from groq import Groq
from sentence_transformers import SentenceTransformer


class ResumeProcessor:
    """Process resume PDFs: extract text, parse info, generate embeddings"""
    
    def __init__(self, groq_api_key: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize with Groq API key for text extraction and embedding model
        
        Args:
            groq_api_key: Groq API key for resume parsing
            embedding_model: Sentence transformer model name (default: all-MiniLM-L6-v2)
        """
        self.client = Groq(api_key=groq_api_key)
        print(f"🤖 Loading sentence transformer model: {embedding_model}")
        
        try:
            # Try loading with local_files_only first (if already cached)
            self.embedding_model = SentenceTransformer(embedding_model, local_files_only=True)
            print(f"✓ Model loaded from cache")
        except Exception as e:
            print(f"⚠️ Model not in cache, downloading from HuggingFace...")
            try:
                # Try downloading with timeout
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"✓ Model downloaded and loaded successfully")
            except Exception as download_error:
                print(f"❌ Failed to download model: {download_error}")
                print(f"⚠️ Network issue detected. Using fallback: Groq embeddings")
                self.embedding_model = None  # Will use Groq API for embeddings as fallback
                self.use_groq_embeddings = True
        
        self.use_groq_embeddings = self.embedding_model is None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
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
                
                # Clean up text
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = text.strip()
                
                print(f"✓ Extracted {len(text)} characters")
                return text
                
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {e}")
            import traceback
            print(traceback.format_exc())
            return ""
    
    def extract_resume_info(self, resume_text: str, resume_filename: str) -> Dict:
        """Extract structured information from resume text using Groq LLM"""
        
        # Use more text for better extraction
        text_for_extraction = resume_text[:25000] if len(resume_text) > 25000 else resume_text
        
        prompt = f"""You are an expert resume parser. Extract the following information from this resume and return ONLY a valid JSON object with no additional text or markdown.

Required JSON structure:
{{
  "candidate_name": "Extract the full name of the candidate",
  "email": "Extract email address if present, otherwise 'Not specified'",
  "phone": "Extract phone number if present, otherwise 'Not specified'",
  "location": "Extract current location, city, or preferred location, otherwise 'Not specified'",
  "summary": "Write a concise 2-3 sentence professional summary based on the resume content",
  "skills": ["Extract all technical and professional skills as an array"],
  "experience_years": "Estimate total years of professional experience as a number or range (e.g., '5' or '5-7')",
  "education": "Extract highest degree and institution (e.g., 'B.S. Computer Science, MIT')",
  "job_titles": ["Extract current and recent job titles"],
  "key_achievements": "List top 3-5 notable achievements or highlights from the resume"
}}

CRITICAL INSTRUCTIONS:
1. Extract actual information from the resume - do NOT use "Not specified" unless the information is truly absent
2. For candidate_name: Look carefully for names at the top of the resume or in contact information section
3. For skills: Extract ALL relevant skills mentioned (technical, soft skills, tools, languages, frameworks)
4. For experience_years: Calculate from job history dates if available
5. For summary: Create a meaningful summary based on the resume content, don't just say "Not specified"
6. Return ONLY the JSON object with no markdown code blocks, no backticks, no additional text
7. Make sure the JSON is valid and properly formatted

Resume Text:
{text_for_extraction}

Return only the JSON object:"""

        try:
            print("🔍 Calling Groq API to extract resume information...")
            
            # Using Llama 3.1 70B for better extraction quality
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Fast and accurate
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert resume parser that extracts structured information accurately. You always return valid JSON without any markdown formatting or additional text."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=4000,  # Groq allows larger responses
                top_p=0.9
            )
            
            result = response.choices[0].message.content.strip()
            print(f"📥 Received response from Groq ({len(result)} characters)")
            
            # Debug: Show raw response
            print(f"Raw API Response:\n{result[:500]}...")
            
            # Clean the response more aggressively
            # Remove markdown code blocks
            result = re.sub(r'```json\s*', '', result, flags=re.IGNORECASE)
            result = re.sub(r'```\s*', '', result)
            result = result.strip()
            
            # Try to extract JSON if there's extra text
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                result = json_match.group(0)
            
            # Parse JSON
            resume_info = json.loads(result)
            print("✓ Successfully parsed JSON response")
            
            # Generate resume_id
            if resume_info.get('candidate_name') and resume_info['candidate_name'] != 'Not specified':
                name_hash = hashlib.md5(resume_info['candidate_name'].encode()).hexdigest()[:8]
                resume_info['resume_id'] = f"resume-{name_hash}"
            else:
                resume_info['resume_id'] = f"resume-{hashlib.md5(resume_filename.encode()).hexdigest()[:8]}"
            
            # Add full text and metadata
            resume_info['full_text'] = resume_text
            resume_info['filename'] = resume_filename
            resume_info['processed_at'] = datetime.utcnow().isoformat()
            
            # Validate and clean data
            if not isinstance(resume_info.get('skills'), list):
                resume_info['skills'] = []
            if not isinstance(resume_info.get('job_titles'), list):
                resume_info['job_titles'] = []
            
            # Log extracted info
            print(f"✓ Extracted Info:")
            print(f"  - Name: {resume_info.get('candidate_name', 'N/A')}")
            print(f"  - Email: {resume_info.get('email', 'N/A')}")
            print(f"  - Skills: {len(resume_info.get('skills', []))} found")
            print(f"  - Experience: {resume_info.get('experience_years', 'N/A')} years")
            
            return resume_info
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Failed to parse response: {result[:500]}...")
            return self._create_fallback_resume_info(resume_text, resume_filename, 
                                                     f"JSON parsing failed: {str(e)}")
            
        except Exception as e:
            print(f"❌ Error extracting resume info: {e}")
            import traceback
            print(traceback.format_exc())
            return self._create_fallback_resume_info(resume_text, resume_filename, str(e))
    
    def _create_fallback_resume_info(self, resume_text: str, resume_filename: str, error_msg: str) -> Dict:
        """Create a fallback resume info dict when extraction fails"""
        print(f"⚠️ Using fallback extraction due to: {error_msg}")
        
        # Try basic regex extraction as fallback
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
        phone_match = re.search(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b', resume_text)
        
        # Extract potential name (usually in first 200 chars)
        first_lines = resume_text[:200].split('\n')
        potential_name = first_lines[0].strip() if first_lines else "Not specified"
        
        # Try to extract skills using common keywords
        skills = []
        common_skills = ['python', 'java', 'javascript', 'react', 'sql', 'aws', 'docker', 
                        'kubernetes', 'machine learning', 'ai', 'data science', 'excel',
                        'tensorflow', 'pytorch', 'node.js', 'angular', 'vue', 'git']
        for skill in common_skills:
            if skill.lower() in resume_text.lower():
                skills.append(skill.capitalize())
        
        return {
            "resume_id": f"resume-{hashlib.md5(resume_filename.encode()).hexdigest()[:8]}",
            "candidate_name": potential_name,
            "email": email_match.group(0) if email_match else "Not specified",
            "phone": phone_match.group(0) if phone_match else "Not specified",
            "location": "Not specified",
            "summary": f"Resume extracted but detailed parsing encountered an issue. Basic information available.",
            "skills": skills,
            "experience_years": "Not specified",
            "education": "Not specified",
            "job_titles": [],
            "key_achievements": "Please review resume manually for detailed achievements",
            "full_text": resume_text,
            "filename": resume_filename,
            "processed_at": datetime.utcnow().isoformat(),
            "extraction_error": error_msg
        }
    
    def generate_embedding(self, text: str) -> Optional[list]:
        """Generate embedding for text using sentence transformers or Groq API"""
        try:
            # Use sentence transformers if available
            if not self.use_groq_embeddings and self.embedding_model is not None:
                max_chars = 50000
                if len(text) > max_chars:
                    text = text[:max_chars]
                
                # Generate embedding
                embedding = self.embedding_model.encode(
                    text,
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
                
                # Convert numpy array to list for JSON serialization
                return embedding.tolist()
            
            # Fallback: Use simple text features (TF-IDF-like approach)
            else:
                print("⚠️ Using fallback embedding method (text features)")
                return self._generate_simple_embedding(text)
            
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            print("⚠️ Attempting fallback embedding method...")
            return self._generate_simple_embedding(text)
    
    def _generate_simple_embedding(self, text: str, dimensions: int = 384) -> list:
        """
        Generate a simple embedding using text statistics (fallback method)
        This creates a fixed-size vector based on text characteristics
        """
        import numpy as np
        from collections import Counter
        
        # Normalize text
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Create embedding based on various text features
        embedding = np.zeros(dimensions)
        
        # Feature 1: Word frequency (first 100 dims)
        word_freq = Counter(words)
        common_words = word_freq.most_common(100)
        for idx, (word, freq) in enumerate(common_words[:100]):
            if idx < 100:
                embedding[idx] = freq / len(words) if words else 0
        
        # Feature 2: Character n-grams (next 100 dims)
        char_bigrams = [text[i:i+2] for i in range(len(text)-1)]
        bigram_freq = Counter(char_bigrams)
        common_bigrams = bigram_freq.most_common(100)
        for idx, (bigram, freq) in enumerate(common_bigrams[:100]):
            if idx < 100:
                embedding[100 + idx] = freq / len(char_bigrams) if char_bigrams else 0
        
        # Feature 3: Skill keywords (next 84 dims)
        skills = ['python', 'java', 'javascript', 'react', 'sql', 'aws', 'docker', 
                 'kubernetes', 'machine learning', 'ai', 'data', 'science', 'engineer',
                 'developer', 'manager', 'senior', 'lead', 'architect', 'analyst',
                 'design', 'test', 'qa', 'devops', 'cloud', 'api', 'frontend', 'backend',
                 'full stack', 'mobile', 'web', 'database', 'linux', 'windows', 'git',
                 'agile', 'scrum', 'ci/cd', 'security', 'network', 'project', 'team',
                 'leadership', 'communication', 'problem solving', 'analytics', 'tableau',
                 'powerbi', 'excel', 'office', 'html', 'css', 'node', 'angular', 'vue',
                 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'scala',
                 'spark', 'hadoop', 'kafka', 'redis', 'mongodb', 'postgresql', 'mysql',
                 'azure', 'gcp', 'terraform', 'ansible', 'jenkins', 'jira', 'confluence',
                 'rest', 'graphql', 'microservices', 'tensorflow', 'pytorch', 'nlp', 'cv']
        
        for idx, skill in enumerate(skills[:84]):
            if idx < 84:
                embedding[200 + idx] = 1.0 if skill in text else 0.0
        
        # Feature 4: Text statistics (remaining dims)
        if len(words) > 0:
            embedding[284] = len(words) / 1000  # Word count normalized
            embedding[285] = len(set(words)) / len(words)  # Unique word ratio
            embedding[286] = sum(len(w) for w in words) / len(words)  # Avg word length
            embedding[287] = text.count('@')  # Email indicators
            embedding[288] = text.count('.')  # Sentence indicators
            embedding[289] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        print(f"✓ Generated fallback embedding ({dimensions} dimensions)")
        return embedding.tolist()
    
    def process_resume_pdf(self, pdf_path: str) -> Optional[Dict]:
        """Process a resume PDF: extract, parse, embed"""
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")
        
        # Extract text
        resume_text = self.extract_text_from_pdf(pdf_path)
        if not resume_text:
            print("❌ No text extracted from PDF")
            return None
        
        # Extract structured info
        print("\n🔍 Extracting structured information...")
        resume_filename = os.path.basename(pdf_path)
        resume_info = self.extract_resume_info(resume_text, resume_filename)
        
        # Generate embedding
        print("\n🧠 Generating embedding...")
        embedding = self.generate_embedding(resume_info['full_text'])
        
        if not embedding:
            print("❌ Failed to generate embedding")
            return None
        
        resume_info['resume_embedding'] = embedding
        resume_info['embedding_created_at'] = datetime.utcnow().isoformat()
        print(f"✓ Embedding created ({len(embedding)} dimensions)")
        
        print(f"\n{'='*60}")
        print("✅ Resume processing complete!")
        print(f"{'='*60}\n")
        
        return resume_info