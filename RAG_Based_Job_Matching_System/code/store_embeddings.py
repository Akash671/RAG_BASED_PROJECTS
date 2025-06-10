import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def store_embeddings(cv_data):
    if not cv_data:  
        raise ValueError("❌ Error: No CV text found!")

    texts = list(cv_data.values())
    cv_ids = list(cv_data.keys())

    embeddings = sentence_model.encode(texts, convert_to_numpy=True)

    if not embeddings.any():
        raise ValueError("❌ Error: No embeddings generated!")

    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    return index, cv_ids
