#!/usr/bin/env python3

import os # Ensure os is imported at the beginning
import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import kagglehub # Import kagglehub
import faiss # Import faiss
from sentence_transformers import SentenceTransformer # Import SentenceTransformer

# --- Loading Functions (using job_matcher's functions) ---
# Use Streamlit caching for efficiency

@st.cache_data
def load_data():
    """Loads, filters, and preprocesses the job dataset by downloading from Kaggle."""
    try:
        st.info("Downloading dataset from Kaggle using kagglehub...")
        path = kagglehub.dataset_download("batuhanmutlu/job-skill-set")
        csv_file_path = os.path.join(path, "all_job_post.csv") # Assuming the CSV name is still all_job_post.csv after download
        st.success(f"Dataset downloaded to: {csv_file_path}")

        df = pd.read_csv(csv_file_path)
        # Apply the same filtering and preprocessing steps as in the notebook
        # Exclude 'job_title' from dropna as it's missing in the updated dataset structure
        df = df.dropna(subset=["job_description", "job_skill_set", "category"])
        # Skip filtering by job_title counts as the column is missing
        # job_title_counts = df['job_title'].value_counts()
        # to_remove = job_title_counts[job_title_counts == 1].index
        # df = df[~df['job_title'].isin(to_remove)]
        # Combine description + skills
        df["full_text"] = df["job_description"] + " " + df["job_skill_set"]
        st.success(f"Dataset loaded and preprocessed. Found {len(df)} job entries.")
        return df
    except ImportError:
         st.error("Kagglehub not installed. Cannot download dataset automatically. Please install it: pip install kagglehub")
         st.stop()
    except Exception as e:
        st.error(f"Error loading or processing dataset: {e}")
        st.stop()


# Load the embedding model using the function from job_matcher
@st.cache_resource
def load_embedding_model_st():
    """Loads the sentence-transformers embedding model with Streamlit caching."""
    try:
        # Import load_embedding_model from job_matcher
        from job_matcher import load_embedding_model
        model = load_embedding_model()
        if model:
            st.success("Embedding model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# Load the FAISS index using the function from job_matcher
@st.cache_resource
def load_faiss_index_st(index_file="faiss_index.bin"):
    """Loads the FAISS index with Streamlit caching."""
    try:
        # Import load_faiss_index from job_matcher
        from job_matcher import load_faiss_index
        index = load_faiss_index(index_file)
        if index:
            st.success(f"FAISS index loaded successfully from {index_file}.")
        return index
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

# --- Load Model, Index, and Data ---
# Load data using the existing cached function
df = load_data()
# Load embedding model and FAISS index using the new cached functions
embedding_model = load_embedding_model_st()
faiss_index = load_faiss_index_st()


# --- Streamlit App Layout ---
st.title("Job Matching App (FAISS Search)") # Updated title

user_skills = st.text_input("Enter your skills (comma-separated):")
user_predicted_sector = st.text_input("Enter your predicted job sector (optional):")

if st.button("Find Similar Jobs"):
    # Combine skills and predicted sector
    user_input = user_skills.strip()
    if user_predicted_sector:
        user_input = f"{user_input} {user_predicted_sector.strip()}"

    if not user_input:
        st.warning("Please enter your skills to find similar jobs.")
    elif df is None or embedding_model is None or faiss_index is None:
         st.error("Data, model, or FAISS index not loaded. Cannot perform search.")
    else:
        st.info("Finding similar job entries using FAISS index...")
        try:
            # Import the FAISS search function from job_matcher
            from job_matcher import find_similar_jobs_faiss
            # Call the FAISS-based search function
            top_similar_job_indices = find_similar_jobs_faiss(user_input, df, embedding_model, faiss_index, top_n=5)

            # --- Display Results ---
            # Check if 'job_title' column exists before attempting to display job titles
            if 'job_title' in df.columns:
                st.subheader("Top 5 Similar Job Titles:")
                if top_similar_job_indices:
                    for index in top_similar_job_indices:
                        # Safely access job_title using .loc
                        if index in df.index:
                            job_title = df.loc[index, 'job_title']
                            st.write(f"- {job_title}")
                        else:
                            st.write(f"- Index {index}: Job title not available (index not found in DataFrame).") # Fallback

                else:
                    st.write("No similar job titles found.")
            else:
                 # Fallback if 'job_title' column is not available
                st.subheader("Top 5 Similar Job Entries (based on index):")
                if top_similar_job_indices:
                    for index in top_similar_job_indices:
                        # Safely access description using .loc
                        if index in df.index:
                            description = df.loc[index, 'job_description']
                            st.write(f"- Index {index}: {description[:200]}...") # Print first 200 chars
                        else:
                            st.write(f"- Index {index}: Description not available (index not found in DataFrame).")
                else:
                    st.write("No similar job entries found.")


        except Exception as e:
            st.error(f"An error occurred while finding similar jobs: {e}")


# Add instructions for running (already present from previous steps)
# Keeping this here for completeness of the script structure

"""
# How to Run the Job Matching Streamlit Application

1.  **Save the script:** Save this file as `streamlit_app.py`.
2.  **Ensure `job_matcher.py` and `faiss_index.bin` are accessible:** Make sure the `job_matcher.py` file and the `faiss_index.bin` file (created in previous steps) are in the same directory as `streamlit_app.py`, or in your Python path.
3.  **Ensure `all_job_post.csv` is accessible:** The `load_data` function in `job_matcher.py` attempts to download the dataset using `kagglehub`. Ensure `kagglehub` is installed (`pip install kagglehub`) and configured if necessary, or manually place the `all_job_post.csv` file in a location accessible to the script.
4.  **Install dependencies:** If you haven't already, install the required Python libraries by running the following command in your terminal:
"""
