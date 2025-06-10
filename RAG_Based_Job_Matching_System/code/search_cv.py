from preprocess_cv import preprocess_text
from sentence_transformers import SentenceTransformer
import numpy as np

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_top_cv(job_description, faiss_index, cv_ids):
    job_desc_processed = preprocess_text(job_description)
    job_vector = sentence_model.encode(job_desc_processed, convert_to_numpy=True).reshape(1, -1)

    distances, indices = faiss_index.search(job_vector, k=5)

    return [cv_ids[idx] for idx in indices[0]]  # Return CV IDs
