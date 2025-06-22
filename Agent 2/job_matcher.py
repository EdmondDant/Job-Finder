
import os
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import kagglehub # Import kagglehub
import faiss # Import faiss
from sentence_transformers import SentenceTransformer # Import SentenceTransformer

# --- Loading Functions with Caching (for potential use in Streamlit) ---
# These can also be used within the main script for consistency

def load_embedding_model():
    """Loads the sentence-transformers embedding model."""
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("Embedding model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading the embedding model: {e}")
        return None

def load_faiss_index(index_file="faiss_index.bin"):
    """Loads the FAISS index from a local file."""
    try:
        if os.path.exists(index_file):
            index = faiss.read_index(index_file)
            print(f"FAISS index loaded successfully from {index_file}. Number of vectors: {index.ntotal}")
            return index
        else:
            print(f"Error: FAISS index file not found at {index_file}. Please ensure the index is created and saved.")
            return None
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def load_data():
    """Loads, filters, and preprocesses the job dataset by downloading from Kaggle."""
    try:
        print("Downloading dataset from Kaggle using kagglehub...")
        path = kagglehub.dataset_download("batuhanmutlu/job-skill-set")
        csv_file_path = os.path.join(path, "all_job_post.csv") # Assuming the CSV name is still all_job_post.csv after download
        print(f"Dataset downloaded to: {csv_file_path}")

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
        print(f"Dataset loaded and preprocessed. Found {len(df)} job entries.")
        return df
    except ImportError:
         print("Kagglehub not installed. Cannot download dataset automatically. Please install it: pip install kagglehub")
         return None
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return None


# Function to find top N similar job entries using FAISS index
def find_similar_jobs_faiss(user_input, df, embedding_model, faiss_index, top_n=5):
    """
    Finds top N similar job entries using a FAISS index.

    Args:
        user_input (str): The user's skills and/or predicted sector.
        df (pd.DataFrame): The DataFrame containing job data.
        embedding_model: The loaded sentence-transformers embedding model.
        faiss_index: The loaded FAISS index.
        top_n (int): The number of top similar jobs to return.

    Returns:
        list: A list of DataFrame indices of the top similar job entries.
    """
    if embedding_model is None or faiss_index is None:
        print("Embedding model or FAISS index not loaded. Cannot perform search.")
        return []

    try:
        # Encode the user input
        user_embedding = embedding_model.encode([user_input])

        # Perform similarity search using FAISS
        distances, indices = faiss_index.search(user_embedding, top_n)

        # The indices returned by FAISS correspond to the row indices in the original embeddings array.
        # Since the embeddings were created from df['full_text'].tolist(),
        # these indices should directly correspond to the DataFrame indices.
        # However, it's safer to map these indices back to the original DataFrame index.
        # If the DataFrame was not filtered/sampled in a way that changes the index order
        # relative to the embeddings, the FAISS indices should align with df.index.

        # Returning the indices from the FAISS search directly.
        # This assumes the order of embeddings added to the index
        # matches the order of rows in the DataFrame used to create them.
        # If the DataFrame was sampled or reordered before creating embeddings,
        # a mapping would be needed. Given the notebook's flow, they should align.
        top_faiss_indices = indices[0].tolist()

        # Map FAISS indices back to original DataFrame indices if necessary
        # For this case, assuming direct mapping is sufficient
        top_df_indices = [df.index[i] for i in top_faiss_indices]


        return top_df_indices

    except Exception as e:
        print(f"An error occurred during FAISS search: {e}")
        return []


if __name__ == "__main__":
    # --- Load Model, Index, and Data ---
    embedding_model = load_embedding_model()
    faiss_index = load_faiss_index()
    df = load_data()

    if embedding_model is None or faiss_index is None or df is None:
        print("Failed to load necessary components. Exiting.")
        exit()

    # --- Get User Input ---
    user_skills = input("Enter your skills (comma-separated): ")
    user_predicted_sector = input("Enter your predicted job sector (optional, press Enter to skip): ")

    user_input = user_skills.strip()
    if user_predicted_sector:
        user_input = f"{user_input} {user_predicted_sector.strip()}"

    if not user_input:
        print("No input provided. Exiting.")
        exit()

    # --- Find Similar Jobs ---
    print("\nFinding similar job entries...")
    try:
        # Use the FAISS-based search function
        top_similar_job_indices = find_similar_jobs_faiss(user_input, df, embedding_model, faiss_index, top_n=5)

        # --- Print Results ---
        print("\nTop 5 Similar Job Entries (based on index):")
        if top_similar_job_indices:
            for index in top_similar_job_indices:
                 # Safely access description using .loc
                # Check if index exists in DataFrame before accessing
                if index in df.index:
                    description = df.loc[index, 'job_description']
                    print(f"- Index {index}: {description[:200]}...") # Print first 200 chars
                else:
                    print(f"- Index {index}: Description not available (index not found in DataFrame).")
        else:
            print("No similar job entries found.")

    except Exception as e:
        print(f"An error occurred while finding similar jobs: {e}")
