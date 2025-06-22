
import os
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
# Remove this line as ktrain is not needed for the embedding approach

# Function to get embedding for a given text
def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    outputs = model(**inputs)
    # Use the mean of the last hidden states as the embedding
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to find top N similar job titles
def find_similar_jobs(user_input, job_texts, job_titles, model, tokenizer, top_n=5):
    user_embedding = get_embedding(user_input, model, tokenizer)
    # Ensure job_embeddings are calculated within the script's execution
    job_embeddings = np.array([get_embedding(text, model, tokenizer)[0] for text in job_texts])

    # Calculate cosine similarity between user input and job descriptions
    similarities = cosine_similarity(user_embedding.reshape(1, -1), job_embeddings)

    # Get the indices of top N most similar jobs
    top_indices = similarities[0].argsort()[-top_n:][::-1]

    # Get the corresponding job titles
    top_job_titles = [job_titles[i] for i in top_indices]

    return top_job_titles

if __name__ == "__main__":
    # --- Setup: Install libraries (if not already present) ---
    # This part assumes users might not have the libraries installed.
    # In a production environment, you'd handle dependencies differently.
    # try:
    #     import pandas as pd
    #     from transformers import AutoModel, AutoTokenizer
    #     from sklearn.metrics.pairwise import cosine_similarity
    #     import numpy as np
    # except ImportError:
    #     print("Installing required libraries...")
    #     os.system("pip install pandas transformers scikit-learn numpy")
    #     import pandas as pd
    #     from transformers import AutoModel, AutoTokenizer
    #     from sklearn.metrics.pairwise import cosine_similarity
    #     import numpy as np


    # --- Load Model and Tokenizer ---
    # Load the pre-trained model and tokenizer for embeddings
    try:
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        exit()

    # --- Load Data ---
    # You'll need to provide the path to the CSV file.
    # This assumes the CSV file is accessible to the user running the script.
    # If the dataset is large, you might consider sharing a preprocessed
    # version or embeddings directly.
    csv_file_path = "all_job_post.csv" # *** Update this path ***

    if not os.path.exists(csv_file_path):
        print(f"Error: Dataset not found at {csv_file_path}. Please provide the correct path or download the dataset.")
        # As an alternative, you could include code to download the dataset
        # using kagglehub if the user has it installed and configured.
        # try:
        #     import kagglehub
        #     print("Attempting to download dataset using kagglehub...")
        #     path = kagglehub.dataset_download("batuhanmutlu/job-skill-set")
        #     csv_file_path = os.path.join(path, "all_job_post.csv")
        #     print(f"Dataset downloaded to: {csv_file_path}")
        # except ImportError:
        #      print("Kagglehub not installed. Cannot download dataset automatically.")
        # except Exception as e:
        #      print(f"Error downloading dataset with kagglehub: {e}")
        exit()


    try:
        df = pd.read_csv(csv_file_path)
        # Apply the same filtering and preprocessing steps as in the notebook
        df = df.dropna(subset=["job_description", "job_title", "job_skill_set", "category"])
        job_title_counts = df['job_title'].value_counts()
        to_remove = job_title_counts[job_title_counts == 1].index
        df = df[~df['job_title'].isin(to_remove)]
        # Combine description + skills
        df["full_text"] = df["job_description"] + " " + df["job_skill_set"]
        # Create text lists for embedding
        job_texts = df["full_text"].tolist()
        job_titles = df["job_title"].tolist()
        print(f"Dataset loaded and preprocessed. Found {len(job_texts)} job entries.")

    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        exit()


    # --- Get User Input ---
    user_skills = input("Enter your skills (comma-separated): ")
    user_predicted_sector = input("Enter your predicted job sector (optional, press Enter to skip): ")

    user_input = user_skills
    if user_predicted_sector:
        user_input = f"{user_skills} {user_predicted_sector}"

    if not user_input.strip():
        print("No input provided. Exiting.")
        exit()

    # --- Find Similar Jobs ---
    print("\nFinding similar job titles...")
    try:
        top_similar_jobs = find_similar_jobs(user_input, job_texts, job_titles, model, tokenizer, top_n=5)

        # --- Print Results ---
        print("\nTop 5 similar job titles:")
        if top_similar_jobs:
            for job in top_similar_jobs:
                print(f"- {job}")
        else:
            print("No similar job titles found.")

    except Exception as e:
        print(f"An error occurred while finding similar jobs: {e}")
