from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_and_embed(cv_data):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    cv_chunks = {cv_id: text_splitter.split_text(text) for cv_id, text in cv_data.items()}

    return {cv_id: sentence_model.encode(texts, convert_to_numpy=True) for cv_id, texts in cv_chunks.items()}
