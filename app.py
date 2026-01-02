"""
Streamlit UI for Job Search & Resume Matching System
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.append('src')

from job_embedding_pipeline import JobEmbeddingPipeline
from resume_matching_pipeline import ResumeMatchingPipeline
from elasticsearch_manager import ElasticsearchManager

# Page config
st.set_page_config(
    page_title="Job Search & Resume Matcher",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.es_connected = False
    st.session_state.groq_api_key = os.getenv('GROQ_API_KEY', '')
    st.session_state.es_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
    st.session_state.resume_processed = False
    st.session_state.current_resume_id = None

# Auto-connect on startup
if not st.session_state.es_connected and st.session_state.groq_api_key and st.session_state.es_url:
    try:
        es_manager = ElasticsearchManager(st.session_state.es_url)
        if es_manager.ping():
            st.session_state.es_connected = True
    except:
        pass

# Sidebar - Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key input (if not in environment)
    if not st.session_state.groq_api_key:
        st.warning("⚠️ Groq API Key not found")
        groq_key_input = st.text_input(
            "Enter Groq API Key",
            type="password",
            help="Get your API key from https://console.groq.com"
        )
        if groq_key_input:
            st.session_state.groq_api_key = groq_key_input
            st.rerun()
    else:
        st.success("✓ Groq API Key configured")
    
    # Show connection status
    if st.session_state.es_connected:
        st.success("🟢 Elasticsearch Connected")
        
        # Index stats
        st.divider()
        st.subheader("📊 Index Statistics")
        
        es_manager = ElasticsearchManager(st.session_state.es_url)
        
        try:
            job_stats = es_manager.get_index_stats("job_embeddings")
            st.metric("Jobs Indexed", job_stats.get('document_count', 0))
        except:
            st.metric("Jobs Indexed", "N/A")
    else:
        st.error("🔴 Elasticsearch Not Connected")
        st.info("Please check:\n- Elasticsearch is running\n- URL is correct in .env")
    
    # Show model info
    st.divider()
    st.caption("🤖 Using Groq API")
    st.caption("Model: llama-3.3-70b-versatile")
    st.caption("Embeddings: all-MiniLM-L6-v2")

# Main content
st.title("🎯 Job Search & Resume Matching System")
st.markdown("### AI-Powered Job Discovery and Resume-to-Job Matching")
st.caption("Powered by Groq API (Llama 3.1) & Elasticsearch")

if not st.session_state.groq_api_key:
    st.error("❌ Groq API Key is required")
    st.info("""
    **To get started:**
    1. Get your free Groq API key from: https://console.groq.com
    2. Add it to your `.env` file: `GROQ_API_KEY=your-key-here`
    3. Or enter it in the sidebar
    """)
    st.stop()

if not st.session_state.es_connected:
    st.warning("⚠️ Elasticsearch is not connected")
    st.code("""
# .env file should contain:
GROQ_API_KEY=your-groq-api-key-here
ELASTICSEARCH_URL=http://localhost:9200
    """)
    st.info("You can still test resume extraction without Elasticsearch connection")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📋 Process Jobs",
    "🎯 Find Job Matches",
    "🔍 Search Jobs"
])

# Tab 1: Process Jobs
with tab1:
    st.header("📋 Process Job Listings")
    
    if not st.session_state.es_connected:
        st.warning("⚠️ Elasticsearch connection required for this feature")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload jobs JSON file",
            type=['json'],
            help="Upload us_jobs.json or similar file"
        )
    
    with col2:
        max_jobs = st.number_input(
            "Max jobs to process",
            min_value=1,
            max_value=1000,
            value=10,
            help="Limit number of jobs for testing"
        )
        
        recreate_index = st.checkbox(
            "Recreate index",
            value=False,
            help="Delete and recreate the index"
        )
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_file = "temp_jobs.json"
        with open(temp_file, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        if st.button("🚀 Process Jobs", type="primary"):
            with st.spinner("Processing jobs..."):
                try:
                    pipeline = JobEmbeddingPipeline(
                        st.session_state.groq_api_key,
                        st.session_state.es_url
                    )
                    
                    count = pipeline.process_and_index_jobs(
                        temp_file,
                        "job_embeddings",
                        max_jobs,
                        recreate_index
                    )
                    
                    st.success(f"✅ Successfully processed and indexed {count} jobs!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        # Clean up
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

# Tab 2: Find Job Matches (Resume Upload)
with tab2:
    st.header("🎯 Upload Resume & Find Matching Jobs")
    
    # Add helpful tips
    with st.expander("ℹ️ Tips for Best Results"):
        st.markdown("""
        - Upload a well-formatted PDF resume
        - Ensure the PDF contains actual text (not scanned images)
        - Include clear contact information at the top
        - List skills and experience explicitly
        - Works best with resumes under 10 pages
        """)
    
    uploaded_resume = st.file_uploader(
        "Upload resume PDF",
        type=['pdf'],
        help="Upload a resume PDF to find matching jobs",
        key="resume_uploader"
    )
    
    if uploaded_resume:
        # Display file info
        st.info(f"📄 File: {uploaded_resume.name} ({uploaded_resume.size:,} bytes)")
        
        # Save temporarily
        temp_pdf = "temp_resume.pdf"
        with open(temp_pdf, 'wb') as f:
            f.write(uploaded_resume.getvalue())
        
        if st.button("🚀 Analyze Resume & Find Jobs", type="primary"):
            # Create an expander for detailed logs
            with st.expander("📋 Processing Logs", expanded=True):
                log_container = st.container()
                
                try:
                    from resume_processor import ResumeProcessor
                    from elasticsearch_manager import ElasticsearchManager
                    
                    with log_container:
                        st.info("🤖 Initializing Groq-powered resume processor...")
                    
                    # Initialize processor with Groq
                    resume_processor = ResumeProcessor(st.session_state.groq_api_key)
                    
                    # Extract text
                    with log_container:
                        st.info("📄 Step 1/4: Extracting text from PDF...")
                    
                    resume_text = resume_processor.extract_text_from_pdf(temp_pdf)
                    
                    if not resume_text:
                        st.error("❌ Failed to extract text from PDF. Please ensure the PDF contains actual text (not scanned images).")
                        st.stop()
                    
                    st.success(f"✓ Extracted {len(resume_text):,} characters from PDF")
                    
                    # Show preview of extracted text
                    with st.expander("📝 Preview Extracted Text (First 500 chars)"):
                        st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
                    
                    # Extract info
                    with log_container:
                        st.info("🔍 Step 2/4: Analyzing resume with Groq AI (Llama 3.1 70B)...")
                    
                    resume_filename = uploaded_resume.name
                    resume_info = resume_processor.extract_resume_info(resume_text, resume_filename)
                    
                    # Check if extraction was successful
                    if 'extraction_error' in resume_info:
                        st.warning(f"⚠️ Extraction had issues: {resume_info['extraction_error']}")
                        st.info("Using fallback extraction with regex patterns...")
                    
                    st.success("✓ Resume analysis complete!")
                    
                    # Display candidate info with better formatting
                    st.divider()
                    st.subheader("👤 Candidate Profile")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        candidate_name = resume_info.get('candidate_name', 'N/A')
                        if candidate_name == 'Not specified':
                            st.metric("Candidate", "⚠️ Not Found", help="Name could not be extracted")
                        else:
                            st.metric("Candidate", candidate_name)
                    
                    with col2:
                        exp_years = resume_info.get('experience_years', 'N/A')
                        st.metric("Experience", f"{exp_years} years" if exp_years != 'Not specified' else "N/A")
                    
                    with col3:
                        skills = resume_info.get('skills', [])
                        st.metric("Skills Found", len(skills))
                    
                    # Show detailed info in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📞 Contact Information:**")
                        email = resume_info.get('email', 'N/A')
                        phone = resume_info.get('phone', 'N/A')
                        location = resume_info.get('location', 'N/A')
                        
                        if email != 'Not specified':
                            st.text(f"📧 {email}")
                        else:
                            st.text("📧 Email: Not found")
                        
                        if phone != 'Not specified':
                            st.text(f"📱 {phone}")
                        else:
                            st.text("📱 Phone: Not found")
                        
                        if location != 'Not specified':
                            st.text(f"📍 {location}")
                        else:
                            st.text("📍 Location: Not found")
                        
                        st.markdown("**🎓 Education:**")
                        education = resume_info.get('education', 'N/A')
                        if education != 'Not specified':
                            st.text(education)
                        else:
                            st.text("Not found in resume")
                    
                    with col2:
                        st.markdown("**💼 Skills:**")
                        if skills:
                            # Display skills as tags
                            skills_text = ", ".join(skills[:20])
                            if len(skills) > 20:
                                skills_text += f" ... (+{len(skills)-20} more)"
                            st.text(skills_text)
                        else:
                            st.text("No skills extracted")
                        
                        st.markdown("**👔 Recent Roles:**")
                        job_titles = resume_info.get('job_titles', [])
                        if job_titles:
                            for idx, title in enumerate(job_titles[:3], 1):
                                st.text(f"{idx}. {title}")
                            if len(job_titles) > 3:
                                st.caption(f"... +{len(job_titles)-3} more roles")
                        else:
                            st.text("No job titles extracted")
                    
                    # Show summary in full width
                    summary = resume_info.get('summary', '')
                    if summary and summary != 'Not specified':
                        st.markdown("**📝 Professional Summary:**")
                        st.info(summary)
                    
                    # Show achievements if available
                    achievements = resume_info.get('key_achievements', '')
                    if achievements and achievements != 'Not specified':
                        st.markdown("**🏆 Key Achievements:**")
                        st.success(achievements)
                    
                    # Generate embedding
                    with log_container:
                        st.info("🧠 Step 3/4: Creating semantic embedding...")
                    
                    resume_embedding = resume_processor.generate_embedding(resume_info['full_text'])
                    
                    if not resume_embedding:
                        st.error("❌ Failed to create embedding")
                        st.stop()
                    
                    st.success(f"✓ Embedding created ({len(resume_embedding)} dimensions)")
                    
                    # Search for similar jobs (only if ES is connected)
                    if st.session_state.es_connected:
                        st.divider()
                        with log_container:
                            st.info("🎯 Step 4/4: Searching for matching jobs...")
                        
                        st.subheader("🎯 Matching Jobs")
                        
                        es_manager = ElasticsearchManager(st.session_state.es_url)
                        
                        # Search using embedding
                        matches = es_manager.semantic_search(
                            resume_embedding,
                            "job_embeddings",
                            "description_embedding",
                            10
                        )
                        
                        if matches:
                            st.success(f"✅ Found {len(matches)} matching jobs!")
                            
                            # Display matches
                            for i, job in enumerate(matches, 1):
                                score = job.get('_score', 0)
                                match_pct = score * 100
                                
                                # Color based on match percentage
                                if match_pct >= 85:
                                    color = "🟢"
                                    label = "Excellent Match"
                                elif match_pct >= 75:
                                    color = "🟡"
                                    label = "Good Match"
                                elif match_pct >= 65:
                                    color = "🟠"
                                    label = "Fair Match"
                                else:
                                    color = "🔴"
                                    label = "Low Match"
                                
                                with st.expander(
                                    f"{color} {i}. {job.get('job_title', 'N/A')} at {job.get('company', 'N/A')} - {label}",
                                    expanded=(i <= 3)
                                ):
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.markdown(f"**Location:** {job.get('location', 'N/A')}")
                                        st.markdown(f"**Salary:** {job.get('salary_range', 'Not specified')}")
                                        st.markdown(f"**Job ID:** `{job.get('job_id', 'N/A')}`")
                                    
                                    with col2:
                                        st.metric("Match", f"{match_pct:.1f}%")
                                        st.progress(match_pct / 100)
                                    
                                    st.markdown("**Role:**")
                                    st.write(job.get('role', 'N/A'))
                                    
                                    st.markdown(f"[🔗 Apply Now]({job.get('link', '#')})")
                        else:
                            st.warning("⚠️ No matches found. Please process some jobs first in the 'Process Jobs' tab.")
                        
                        with log_container:
                            st.success("✅ Processing complete!")
                    else:
                        st.warning("⚠️ Elasticsearch not connected. Resume analyzed but job matching unavailable.")
                        st.info("Connect Elasticsearch to enable job matching feature.")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    with st.expander("🐛 Full Error Details"):
                        st.code(traceback.format_exc())
        
        # Clean up
        if os.path.exists(temp_pdf):
            try:
                os.remove(temp_pdf)
            except:
                pass

# Tab 3: Search Jobs
with tab3:
    st.header("🔍 Search Jobs")
    
    if not st.session_state.es_connected:
        st.warning("⚠️ Elasticsearch connection required for job search")
        st.stop()
    
    search_query = st.text_input(
        "Enter search query",
        placeholder="e.g., Senior Python Developer with ML experience",
        help="Enter keywords or description of the job you're looking for"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_type = st.selectbox(
            "Search Type",
            ["Semantic (AI)", "Keyword", "Hybrid"],
            help="Semantic uses AI understanding, Keyword uses text matching"
        )
    
    with col2:
        num_results = st.slider(
            "Number of results",
            min_value=5,
            max_value=50,
            value=10
        )
    
    with col3:
        if search_type == "Hybrid":
            semantic_weight = st.slider(
                "Semantic weight",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher = more AI, Lower = more keywords"
            )
    
    if st.button("🔍 Search", type="primary") and search_query:
        with st.spinner("Searching..."):
            try:
                pipeline = JobEmbeddingPipeline(
                    st.session_state.groq_api_key,
                    st.session_state.es_url
                )
                
                if search_type == "Semantic (AI)":
                    results = pipeline.search_jobs_semantic(
                        search_query,
                        "job_embeddings",
                        num_results
                    )
                elif search_type == "Keyword":
                    results = pipeline.search_jobs_keyword(
                        search_query,
                        "job_embeddings",
                        num_results
                    )
                else:  # Hybrid
                    results = pipeline.search_jobs_hybrid(
                        search_query,
                        "job_embeddings",
                        num_results,
                        semantic_weight
                    )
                
                if results:
                    st.success(f"Found {len(results)} jobs!")
                    
                    for i, job in enumerate(results, 1):
                        with st.expander(
                            f"{i}. {job.get('job_title', 'N/A')} at {job.get('company', 'N/A')}",
                            expanded=(i <= 3)
                        ):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Location:** {job.get('location', 'N/A')}")
                                st.markdown(f"**Salary:** {job.get('salary_range', 'Not specified')}")
                                st.markdown(f"**Job ID:** `{job.get('job_id', 'N/A')}`")
                            
                            with col2:
                                if 'rrf_score' in job:
                                    st.caption(f"Semantic #{job.get('semantic_rank', 'N/A')}")
                                    st.caption(f"Keyword #{job.get('keyword_rank', 'N/A')}")
                            
                            st.markdown("**Role:**")
                            st.write(job.get('role', 'N/A'))
                            
                            st.markdown(f"[🔗 View Job]({job.get('link', '#')})")
                else:
                    st.warning("No results found")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Footer
st.divider()
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.caption("🎯 Job Search & Resume Matching System")
with col2:
    st.caption("⚡ Powered by Groq (Llama 3.1)")
with col3:
    st.caption("🔍 Elasticsearch")