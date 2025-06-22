import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.feature_extraction.text import TfidfVectorizer
from pdfminer.high_level import extract_text
import io
import os
import kagglehub
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from huggingface_hub import hf_hub_download

# Configure page
st.set_page_config(
    page_title="Job Recommendation System",
    page_icon="💼",
    layout="wide"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'interests'
if 'user_interests' not in st.session_state:
    st.session_state.user_interests = []
if 'recommended_sector' not in st.session_state:
    st.session_state.recommended_sector = ""
if 'user_skills' not in st.session_state:
    st.session_state.user_skills = []
if 'recommended_job' not in st.session_state:
    st.session_state.recommended_job = ""
if 'recommended_jobs_list' not in st.session_state:
    st.session_state.recommended_jobs_list = []
if 'extracted_skills' not in st.session_state:
    st.session_state.extracted_skills = {}
if 'skills_text' not in st.session_state:
    st.session_state.skills_text = ""
if 'extracted_name' not in st.session_state:
    st.session_state.extracted_name = ""

# Data mappings (kept for fallback)
SKILLS_TO_JOBS = {
    # Technology
    "python": "Software Developer",
    "java": "Backend Developer",
    "javascript": "Frontend Developer",
    "data analysis": "Data Analyst",
    "machine learning": "ML Engineer",
    "sql": "Database Administrator",
    "cybersecurity": "Security Analyst",
    "cloud computing": "Cloud Engineer",
    
    # Healthcare
    "patient care": "Registered Nurse",
    "medical diagnosis": "Physician",
    "pharmaceutical": "Pharmacist",
    "medical research": "Medical Researcher",
    
    # Finance
    "financial analysis": "Financial Analyst",
    "accounting": "Accountant",
    "investment": "Investment Advisor",
    "risk management": "Risk Manager",
    
    # Education
    "teaching": "Teacher",
    "curriculum design": "Curriculum Developer",
    "educational technology": "EdTech Specialist",
    
    # Creative
    "graphic design": "Graphic Designer",
    "content writing": "Content Writer",
    "digital marketing": "Digital Marketing Specialist",
    "video editing": "Video Editor",
    
    # Business
    "project management": "Project Manager",
    "business analysis": "Business Analyst",
    "sales": "Sales Representative",
    "human resources": "HR Specialist",
    
    # Engineering
    "cad design": "Design Engineer",
    "structural analysis": "Structural Engineer",
    "quality control": "Quality Engineer",
    
    # Science
    "laboratory research": "Research Scientist",
    "data collection": "Research Associate",
    "environmental monitoring": "Environmental Scientist"
}

# --- FAISS Job Matching Functions (from Agent2.py) ---

@st.cache_data
def load_job_data():
    """Loads, filters, and preprocesses the job dataset by downloading from Kaggle."""
    try:
        # Use st.empty() to create a placeholder for temporary messages
        status_placeholder = st.empty()
        status_placeholder.info("Downloading job dataset from Kaggle...")
        
        path = kagglehub.dataset_download("batuhanmutlu/job-skill-set")
        csv_file_path = os.path.join(path, "all_job_post.csv")
        
        df = pd.read_csv(csv_file_path)
        # Apply filtering and preprocessing
        df = df.dropna(subset=["job_description", "job_skill_set", "category"])
        # Combine description + skills
        df["full_text"] = df["job_description"] + " " + df["job_skill_set"]
        
        # Clear the status message
        status_placeholder.empty()
        
        return df
    except Exception as e:
        st.warning(f"Could not load job dataset from Kaggle: {e}")
        st.info("Using fallback job matching approach...")
        return None

@st.cache_resource
def load_embedding_model():
    """Loads the sentence-transformers embedding model."""
    try:
        # Use st.empty() to create a placeholder for temporary messages
        status_placeholder = st.empty()
        status_placeholder.info("Loading embedding model...")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Clear the status message
        status_placeholder.empty()
        
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

@st.cache_resource
def load_faiss_index(index_file="faiss_index.bin"):
    """Loads the FAISS index."""
    try:
        # Use st.empty() to create a placeholder for temporary messages
        status_placeholder = st.empty()
        status_placeholder.info("Loading FAISS index...")
        
        if os.path.exists(index_file):
            index = faiss.read_index(index_file)
            # Clear the status message
            status_placeholder.empty()
            return index
        else:
            status_placeholder.warning(f"FAISS index file {index_file} not found. Please ensure it exists.")
            return None
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

def find_similar_jobs_faiss(user_input, df, embedding_model, faiss_index, top_n=5):
    """Find similar jobs using FAISS index."""
    try:
        # Generate embedding for user input
        user_embedding = embedding_model.encode([user_input])
        user_embedding = np.array(user_embedding).astype('float32')
        
        # Search using FAISS
        distances, indices = faiss_index.search(user_embedding, top_n)
        
        # Return the indices of similar jobs
        return indices[0].tolist()
    except Exception as e:
        st.error(f"Error in FAISS search: {e}")
        return []

# --- Original Functions ---

@st.cache_resource
def load_skills_vocabulary():
    """Load the skills vocabulary from skills.txt file"""
    try:
        with open("skills.txt", "r", encoding="utf-8") as f:
            skills = [line.strip().lower() for line in f if line.strip()]
        return skills
    except FileNotFoundError:
        st.warning("skills.txt file not found. Using a basic skills list.")
        # Fallback basic skills list
        return [
            "python", "java", "javascript", "sql", "html", "css", "react", "nodejs",
            "machine learning", "data analysis", "tensorflow", "pytorch", "pandas",
            "numpy", "matplotlib", "tableau", "power bi", "excel", "r", "matlab",
            "project management", "leadership", "communication", "teamwork",
            "problem solving", "critical thinking", "marketing", "sales", "finance"
        ]

@st.cache_resource
def initialize_tfidf_vectorizer():
    """Initialize and fit the TF-IDF Vectorizer with skills vocabulary"""
    skills = load_skills_vocabulary()
    vectorizer = TfidfVectorizer(
        vocabulary=skills,
        ngram_range=(1, 3),
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True
    )
    vectorizer.fit(skills)  # Required to initialize the internal IDF scores
    return vectorizer

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    try:
        # Read the uploaded file into bytes
        pdf_bytes = uploaded_file.read()
        
        # Extract text from PDF bytes
        text = extract_text(io.BytesIO(pdf_bytes))
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_name_from_resume(raw_text):
    """Extract the full name from resume text using basic patterns"""
    if not raw_text.strip():
        return ""
    
    try:
        # Split text into lines
        lines = raw_text.strip().split('\n')
        
        # Common patterns for names (usually at the beginning of the CV)
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Capitalized words at start
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+)',       # Two capitalized words
            r'Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # After "Name:"
            r'([A-Z][A-Z\s]+)',  # All caps name
        ]
        
        # Check first few lines for name patterns
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if not line or len(line) < 3:
                continue
                
            for pattern in name_patterns:
                match = re.search(pattern, line)
                if match:
                    potential_name = match.group(1).strip()
                    # Filter out common non-name patterns
                    if (len(potential_name.split()) >= 2 and 
                        not any(word.lower() in ['resume', 'cv', 'curriculum', 'vitae', 'email', 'phone', 'address'] 
                               for word in potential_name.split())):
                        return potential_name
        
        # If no pattern matches, try first line that looks like a name
        for line in lines[:3]:
            line = line.strip()
            words = line.split()
            if (len(words) >= 2 and len(words) <= 4 and
                all(word.replace('.', '').isalpha() for word in words) and
                all(len(word) > 1 for word in words)):
                return line
        
        return ""
        
    except Exception as e:
        st.error(f"Error extracting name from resume: {str(e)}")
        return ""

def extract_skills_from_resume(uploaded_file):
    """Extract skills from uploaded resume using TF-IDF"""
    try:
        # Get the vectorizer
        vectorizer = initialize_tfidf_vectorizer()
        
        # Extract text from the uploaded PDF
        raw_text = extract_text_from_pdf(uploaded_file)
        
        if not raw_text.strip():
            st.error("Could not extract text from the PDF. Please ensure it's a valid PDF file.")
            return {}, ""
        
        # Extract name from raw text
        extracted_name = extract_name_from_resume(raw_text)
        
        # Clean the text for skill extraction
        cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', raw_text).lower()
        
        # Transform text using TF-IDF
        tfidf_matrix = vectorizer.transform([cleaned_text])
        scores = tfidf_matrix.toarray()[0]
        
        # Map non-zero scores to skill names
        extracted_skills = {
            skill: score
            for skill, score in zip(vectorizer.get_feature_names_out(), scores)
            if score > 0
        }
        
        # Sort by TF-IDF score
        sorted_skills = dict(sorted(extracted_skills.items(), key=lambda x: x[1], reverse=True))
        return sorted_skills, extracted_name
        
    except Exception as e:
        st.error(f"Error extracting skills from resume: {str(e)}")
        return {}, ""

@st.cache_resource
def load_career_model_and_vectorstore():
    """Load the fine-tuned career classification model and vectorstore"""
    try:

        # Load FAISS index locally (bypassing .pkl)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("career_vectorstore", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Load model from Hugging Face
        model_path = "diegowlp/career-model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50, temperature=0.0)
        llm = HuggingFacePipeline(pipeline=pipe)

        # Prompt setup
        prompt_template = PromptTemplate(
            input_variables=["user_input", "retrieved_examples"],
            template="""
You are a career advisor AI. Your job is to assign the best career sector based on a user's input.
Here are some labeled examples:
{retrieved_examples}
Now classify this:
"{user_input}"
Respond only in this format:
sector: <name of best-fit sector>"""
        )

        chain = LLMChain(llm=llm, prompt=prompt_template)
        return retriever, chain

    except Exception as e:
        st.error(f"Error loading career classification model: {str(e)}")
        st.info("Please ensure the career model and vectorstore are available on Hugging Face under 'diegowlp/career-model' and 'diegowlp/career-vectorstore'")
        return None, None


def make_examples(docs):
    """Format retrieved documents as examples for the prompt"""
    return "\n".join([
        f'\"{doc.page_content[:300]}\" → sector: {doc.metadata["category"]}'
        for doc in docs
    ])

def classify_career_sector(user_interests, retriever, chain):
    """Classify career sector using the fine-tuned model"""
    if not retriever or not chain:
        return "Technology"  # Default fallback
    
    try:
        # Combine user interests into a single query
        user_query = ", ".join(user_interests)
        
        # Retrieve similar examples
        top_docs = retriever.invoke(user_query)
        examples = make_examples(top_docs)
        
        # Get sector classification
        response = chain.invoke({"user_input": user_query, "retrieved_examples": examples})
        
        # Extract sector from response
        response_text = response["text"].strip()
        if "sector:" in response_text.lower():
            sector = response_text.split(":")[-1].strip()
            return sector.title()
        else:
            return response_text.strip().title()
            
    except Exception as e:
        st.error(f"Error in sector classification: {str(e)}")
        return "Technology"  # Default fallback

@st.cache_resource
def load_cover_letter_model():
    """Load the fine-tuned model for cover letter generation from Hugging Face"""
    try:
        # Load from Hugging Face model hub
        model_path = "diegowlp/jobfinder-coverletter-model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer, model, device

    except Exception as e:
        st.error(f"Error loading cover letter model: {str(e)}")
        st.info("Please ensure the model exists at 'diegowlp/jobfinder-coverletter-model' on Hugging Face.")
        return None, None, None


def generate_cover_letter(tokenizer, model, device, name, company, job_position,
                         skillsets, qualifications, past_experience, current_experience):
    """Generate cover letter using the fine-tuned model"""
    prompt = f"""Write a personalized and professional cover letter based on the following details:
Applicant Name: {name}
Job Title: {job_position}
Company: {company}
Skillsets: {skillsets}
Qualifications: {qualifications}
Past Working Experience: {past_experience}
Current Working Experience: {current_experience}

The letter should be:
- Formal but engaging
- Clearly structured (greeting, introduction, body, closing)
- Aligned with the role and company
- Focused on strengths and fit

Generate a complete, well-formatted cover letter.
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=768,
            temperature=0.8,
            num_return_sequences=1,
            do_sample=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def find_best_job_faiss(skills, predicted_sector=""):
    """Find the best matching jobs using FAISS-based approach"""
    # Load necessary components
    df = load_job_data()
    embedding_model = load_embedding_model()
    faiss_index = load_faiss_index()
    
    if df is None or embedding_model is None or faiss_index is None:
        st.warning("FAISS-based job matching not available. Using fallback method.")
        return find_best_job_fallback(skills)
    
    try:
        # Combine skills and predicted sector
        user_input = ", ".join(skills)
        if predicted_sector:
            user_input = f"{user_input} {predicted_sector}"
        
        # Find similar jobs using FAISS
        similar_job_indices = find_similar_jobs_faiss(user_input, df, embedding_model, faiss_index, top_n=5)
        
        if similar_job_indices:
            # Get job information for the top matches
            recommended_jobs = []
            for idx in similar_job_indices:
                if idx < len(df):
                    job_info = {
                        'index': idx,
                        'category': df.iloc[idx]['category'],
                        'description': df.iloc[idx]['job_description'][:200] + "...",
                        'skills': df.iloc[idx]['job_skill_set']
                    }
                    # Add job_title if available
                    if 'job_title' in df.columns:
                        job_info['title'] = df.iloc[idx]['job_title']
                    else:
                        # If no job_title column, use category as title
                        job_info['title'] = df.iloc[idx]['category']
                    recommended_jobs.append(job_info)
            
            # Return the top job title (not category) and the list
            top_job_title = recommended_jobs[0]['title'] if recommended_jobs else "Software Developer"
            return top_job_title, recommended_jobs
        else:
            return find_best_job_fallback(skills)
            
    except Exception as e:
        st.error(f"Error in FAISS job matching: {e}")
        return find_best_job_fallback(skills)

def find_best_job_fallback(skills):
    """Fallback job matching using the original approach"""
    skill_scores = {}
    
    for skill in skills:
        skill_lower = skill.lower()
        for skill_key, job in SKILLS_TO_JOBS.items():
            if job not in skill_scores:
                skill_scores[job] = 0
            
            # Check if skill matches
            if skill_lower in skill_key or skill_key in skill_lower:
                skill_scores[job] += 1
    
    if skill_scores:
        best_job = max(skill_scores, key=skill_scores.get)
        return best_job, []
    else:
        return "Software Developer", []

def interests_page():
    """Page 1: Interest to Sector Matching"""
    st.title("🎯 Job Recommendation System")
    st.header("Step 1: Tell us about your interests")
    
    # Initialize session state for career retriever and chain if not already done
    if 'career_retriever' not in st.session_state or 'career_chain' not in st.session_state:
        st.session_state.career_retriever, st.session_state.career_chain = load_career_model_and_vectorstore()
    
    # Interest input
    st.subheader("Describe Your Interests")
    interest_description = st.text_area(
        "Tell us about your interests, hobbies, and what you enjoy doing:",
        height=200,
        placeholder="e.g., I love working with data, solving complex problems, and using technology to help people. I enjoy programming, analyzing trends, and building applications that make a difference."
    )
    
    # Use only the main interest description
    all_interests = [interest_description] if interest_description.strip() else []
    
    if st.button("🔍 Find My Recommended Sector", disabled=len(all_interests) == 0):
        with st.spinner("Analyzing your interests with AI..."):
            st.session_state.user_interests = all_interests
            st.session_state.recommended_sector = classify_career_sector(
                all_interests, 
                st.session_state.career_retriever, 
                st.session_state.career_chain
            )
            
            # Display recommendation
            st.success(f"🎉 Based on your interests, we recommend the **{st.session_state.recommended_sector}** sector!")
    
    # Show current recommendation if exists
    if st.session_state.recommended_sector:
        st.info(f"Current recommendation: **{st.session_state.recommended_sector}** sector")
        
        if st.button("➡️ Next: Find Job Position"):
            st.session_state.current_page = 'skills'
            st.rerun()

def skills_page():
    """Page 2: Skills to Job Matching with CV Upload Option and FAISS-based matching"""
    st.title("💼 Job Position Matching")
    st.header("Step 2: Tell us about your skills")
    
    # Show previous recommendation
    if st.session_state.recommended_sector:
        st.info(f"Recommended sector: **{st.session_state.recommended_sector}**")
    
    # CV Upload section
    st.subheader("📄 Upload CV (Optional)")
    uploaded_file = st.file_uploader(
        "Upload your CV/Resume to automatically extract skills (PDF format)",
        type=['pdf'],
        help="Upload a PDF file of your CV/Resume. We'll automatically extract your skills and populate the skills field below."
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        if st.button("🔍 Extract Skills from CV"):
            with st.spinner("Extracting skills from your CV... This may take a moment."):
                # Extract skills and name from the uploaded CV
                extracted_skills, extracted_name = extract_skills_from_resume(uploaded_file)
                
                if extracted_skills:
                    st.session_state.extracted_skills = extracted_skills
                    st.session_state.extracted_name = extracted_name
                    
                    # Convert extracted skills to a formatted string for the text area
                    skills_list = list(extracted_skills.keys())
                    st.session_state.skills_text = "\n".join([skill.title() for skill in skills_list])
                    
                    st.success(f"✅ Successfully extracted {len(extracted_skills)} skills from your CV!")
                    if extracted_name:
                        st.success(f"✅ Name extracted: **{extracted_name}**")
                    st.info("📝 Skills have been populated in the text area below. You can edit them if needed.")
                    
                    # Force a rerun to update the text area
                    st.rerun()
                else:
                    st.error("Could not extract skills from the CV. Please try manual input or check if the PDF is readable.")
    
    # Skills input section
    st.subheader("📝 Your Skills")
    skills_input = st.text_area(
        "Enter your skills (one per line):",
        value=st.session_state.skills_text,
        height=200,
        placeholder="e.g.\nPython\nSQL\nProject Management\nData Analysis\nLeadership\nCommunication\nProblem Solving",
        help="You can manually enter your skills or use the CV upload above to automatically populate this field."
    )
    
    # Convert skills text to list
    skills_list = [skill.strip() for skill in skills_input.split('\n') if skill.strip()]
    
    # Update session state
    if skills_input != st.session_state.skills_text:
        st.session_state.skills_text = skills_input
    
    # Find job button
    if st.button("🎯 Find My Recommended Jobs", disabled=len(skills_list) == 0):
        with st.spinner("Analyzing your skills using AI-powered job matching..."):
            st.session_state.user_skills = skills_list
            
            # Use FAISS-based job matching
            result = find_best_job_faiss(skills_list, st.session_state.recommended_sector)
            
            if isinstance(result, tuple) and len(result) == 2:
                recommended_job, recommended_jobs_list = result
                st.session_state.recommended_job = recommended_job
                st.session_state.recommended_jobs_list = recommended_jobs_list
            else:
                st.session_state.recommended_job = result
                st.session_state.recommended_jobs_list = []
            
            # Display recommendation - show job title instead of category
            st.success(f"🎉 Based on your skills, here are your top job matches!")
    
    # Show job selection if recommendations exist
    if st.session_state.recommended_jobs_list:
        st.subheader("🎯 Choose Your Preferred Job")
        
        # Create radio button options for job selection
        job_options = []
        job_details = []
        
        for i, job in enumerate(st.session_state.recommended_jobs_list):
            option_text = f"{job['title']}"
            job_options.append(option_text)
            job_details.append(job)
        
        # Add radio button for job selection
        selected_job_index = st.radio(
            "Select the job position that interests you most:",
            range(len(job_options)),
            format_func=lambda x: f"🎯 {job_options[x]}",
            key="job_selection"
        )
        
        # Show details of the selected job
        if selected_job_index is not None:
            selected_job = job_details[selected_job_index]
            
            # Update the recommended job in session state
            st.session_state.recommended_job = selected_job['title']
            
            # Display selected job details
            with st.expander("📋 Selected Job Details", expanded=True):
                st.write(f"**Job Title:** {selected_job['title']}")
                st.write(f"**Required Skills:** {selected_job['skills']}")
        
        # Show why this matches your skills
        with st.expander("🔍 See why these jobs match your skills"):
            st.write("**Your skills:**")
            for skill in st.session_state.user_skills:
                st.write(f"• {skill}")
            
            if st.session_state.recommended_sector:
                st.write(f"**Your predicted sector:** {st.session_state.recommended_sector}")
        
        # Show current selection
        st.info(f"**Selected job:** {st.session_state.recommended_job}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Interests"):
                st.session_state.current_page = 'interests'
                st.rerun()
        with col2:
            if st.button("➡️ Next: Generate Cover Letter"):
                st.session_state.current_page = 'cover_letter'
                st.rerun()
    
    elif st.session_state.recommended_job:
        # Fallback for when we have a recommendation but no detailed list
        st.info(f"Recommended job: **{st.session_state.recommended_job}**")
        
        with st.expander("See why this matches your skills"):
            st.write("**Your skills:**")
            for skill in st.session_state.user_skills:
                st.write(f"• {skill}")
            
            if st.session_state.recommended_sector:
                st.write(f"**Your predicted sector:** {st.session_state.recommended_sector}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Interests"):
                st.session_state.current_page = 'interests'
                st.rerun()
        with col2:
            if st.button("➡️ Next: Generate Cover Letter"):
                st.session_state.current_page = 'cover_letter'
                st.rerun()
    
    else:
        # Show navigation back button if no recommendation yet
        if st.button("⬅️ Back to Interests"):
            st.session_state.current_page = 'interests'
            st.rerun()

def cover_letter_page():
    """Page 3: Cover Letter Generation"""
    st.title("📝 Cover Letter Generator")
    st.header("Step 3: Generate your personalized cover letter")
    
    # Show previous recommendations
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.recommended_sector:
            st.info(f"Recommended sector: **{st.session_state.recommended_sector}**")
    with col2:
        if st.session_state.recommended_job:
            st.info(f"Recommended job: **{st.session_state.recommended_job}**")
    
    st.write("Fill in the details below to generate your cover letter:")
    
    # Cover letter form
    with st.form("cover_letter_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Use extracted name if available, otherwise empty
            default_name = st.session_state.extracted_name if st.session_state.extracted_name else ""
            name = st.text_input("Your Name*", value=default_name, placeholder="John Doe")
            
            company = st.text_input("Company Name*", placeholder="Tech Corp Inc.")
            job_position = st.text_input(
                "Job Position*", 
                value=st.session_state.recommended_job,
                placeholder="Software Developer"
            )
            qualifications = st.text_area(
                "Your Qualifications*",
                placeholder="Bachelor's in Computer Science, 3+ years experience...",
                height=100
            )
        
        with col2:
            # Use ALL user skills, not just the first 5
            all_skills_text = ", ".join(st.session_state.user_skills) if st.session_state.user_skills else ""
            skillsets = st.text_area(
                "Key Skillsets*",
                value=all_skills_text,
                placeholder="Python, SQL, Machine Learning, Problem Solving...",
                height=100,
                help=f"All {len(st.session_state.user_skills)} skills from your profile have been loaded. You can edit them if needed."
            )
        
        past_experience = st.text_area(
            "Past Work Experience*",
            placeholder="Worked as Junior Developer at XYZ Company for 2 years...",
            height=100
        )
        
        current_experience = st.text_area(
            "Current Role/Experience",
            placeholder="Currently working as Software Developer at ABC Corp...",
            height=100
        )
        
        # Show summary of loaded information
        if st.session_state.user_skills or st.session_state.extracted_name:
            with st.expander("📋 Loaded from your profile", expanded=False):
                if st.session_state.extracted_name:
                    st.write(f"**Name:** {st.session_state.extracted_name}")
                if st.session_state.user_skills:
                    st.write(f"**Skills ({len(st.session_state.user_skills)}):** {', '.join(st.session_state.user_skills)}")
                if st.session_state.recommended_sector:
                    st.write(f"**Recommended Sector:** {st.session_state.recommended_sector}")
                if st.session_state.recommended_job:
                    st.write(f"**Recommended Job:** {st.session_state.recommended_job}")
        
        submitted = st.form_submit_button("🚀 Generate Cover Letter")
    
    if submitted:
        # Validate required fields
        required_fields = [name, company, job_position, qualifications, skillsets, past_experience]
        if not all(field.strip() for field in required_fields):
            st.error("Please fill in all required fields marked with *")
        else:
            # Load model and generate cover letter
            with st.spinner("Loading model and generating your cover letter..."):
                tokenizer, model, device = load_cover_letter_model()
                
                if tokenizer and model and device is not None:
                    try:
                        cover_letter = generate_cover_letter(
                            tokenizer, model, device,
                            name, company, job_position,
                            skillsets, qualifications,
                            past_experience, current_experience
                        )
                        
                        st.success("✅ Cover letter generated successfully!")
                        
                        # Display the cover letter
                        st.subheader("Your Generated Cover Letter")
                        st.text_area(
                            "Cover Letter",
                            value=cover_letter,
                            height=400,
                            label_visibility="collapsed"
                        )
                        
                        # Download button
                        filename = f"cover_letter_{name.replace(' ', '_')}_{company.replace(' ', '_')}.txt"
                        st.download_button(
                            label="💾 Download Cover Letter",
                            data=cover_letter,
                            file_name=filename,
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating cover letter: {str(e)}")
                else:
                    st.error("Could not load the model. Please check if the model files are available.")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Skills"):
            st.session_state.current_page = 'skills'
            st.rerun()
    with col2:
        if st.button("🔄 Start Over"):
            # Reset all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.current_page = 'interests'
            st.rerun()

def main():
    """Main application logic"""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Progress indicator
    steps = ["Interests", "Skills", "Cover Letter"]
    current_step = {
        'interests': 0,
        'skills': 1,
        'cover_letter': 2
    }[st.session_state.current_page]
    
    for i, step in enumerate(steps):
        if i < current_step:
            st.sidebar.write(f"✅ {step}")
        elif i == current_step:
            st.sidebar.write(f"🔄 **{step}**")
        else:
            st.sidebar.write(f"⏳ {step}")
    
    # Display current page
    if st.session_state.current_page == 'interests':
        interests_page()
    elif st.session_state.current_page == 'skills':
        skills_page()
    elif st.session_state.current_page == 'cover_letter':
        cover_letter_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("💼 **Job Recommendation System**")
    st.sidebar.markdown("Find your perfect career match!")

if __name__ == "__main__":
    main()