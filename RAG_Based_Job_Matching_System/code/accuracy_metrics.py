from store_embeddings import store_embeddings
from preprocess_cv import preprocess_text
from extract_cv import extract_cv_text, get_cv_id_to_filename
from search_cv import search_top_cv
from langchain.llms import cohere
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")



# Initialize Cohere LLM
COHERE_API_KEY = "cHpZ6GpjYJJEmrlNq2YgTPvrLKaBmwxU1fvxTp7x"
llm = cohere.Cohere(cohere_api_key=COHERE_API_KEY, temperature=0.1)

# Step 1: Extract CVs
cv_data = extract_cv_text()
if not cv_data:
    raise ValueError("❌ Error: CV text extraction failed!")

print("✅ CVs extracted successfully!")

# Step 2: Preprocess CV Texts
preprocessed_cv_data = {cv_id: preprocess_text(text) for cv_id, text in cv_data.items()}

print("✅ CVs preprocessed successfully!")

# Step 3: Store embeddings
vectorstore, cv_ids = store_embeddings(preprocessed_cv_data)

print("✅ CV embeddings stored successfully!")

# Step 4: Get job description
job_description = input("Enter job description: ")
job_desc_processed = preprocess_text(job_description)
job_vector = sentence_model.encode(job_desc_processed, convert_to_numpy=True).reshape(1, -1)

# Step 5: Retrieve matching CVs
top_cv_ids = search_top_cv(job_description, vectorstore, cv_ids)

print("✅ Top CVs retrieved successfully!")

# Step 6: Get file names for matched CVs
file_name_dict = get_cv_id_to_filename()
top_cv_files_name = [file_name_dict.get(cv_id, "Unknown File") for cv_id in top_cv_ids]

print("✅ Top CV file names:", top_cv_files_name)

# Step 7: Measure Accuracy (Precision, Recall, MRR & Cosine Similarity)
def evaluate_accuracy(retrieved_cv_ids, ground_truth, job_vector, preprocessed_cv_data):
    """Calculate Precision, Recall, MRR, and Cosine Similarity for resume retrieval."""
    # Ensure retrieved & ground truth lists are equal in length
    matched_cv_ids = set(retrieved_cv_ids) & set(ground_truth)  # Find common matches
    binary_retrieved = [1 if cv_id in matched_cv_ids else 0 for cv_id in retrieved_cv_ids]
    binary_ground_truth = [1 if cv_id in matched_cv_ids else 0 for cv_id in ground_truth]

    # Make lengths equal for sklearn metrics (pad with zeros if necessary)
    max_len = max(len(binary_ground_truth), len(binary_retrieved))
    binary_ground_truth += [0] * (max_len - len(binary_ground_truth))
    binary_retrieved += [0] * (max_len - len(binary_retrieved))

    # Compute Precision & Recall
    precision = precision_score(binary_ground_truth, binary_retrieved, zero_division=1)
    recall = recall_score(binary_ground_truth, binary_retrieved, zero_division=1)

    # Mean Reciprocal Rank (MRR)
    def mean_reciprocal_rank(retrieved, ground):
        ranks = [1 / (retrieved.index(cv_id) + 1) for cv_id in ground if cv_id in retrieved]
        return np.mean(ranks) if ranks else 0

    mrr = mean_reciprocal_rank(retrieved_cv_ids, ground_truth)

    # Cosine Similarity between job description and retrieved resumes
    resume_embeddings = np.array(
        [sentence_model.encode(preprocessed_cv_data[cv_id], convert_to_numpy=True) for cv_id in retrieved_cv_ids]
    )
    similarities = cosine_similarity(job_vector, resume_embeddings)[0]

    return {
        "Precision": precision,
        "Recall": recall,
        "MRR": mrr,
        "Cosine Similarities": similarities.tolist()
    }


# Define manually verified ground-truth resumes (HR-reviewed)
ground_truth = ["cv_123", "cv_456", "cv_789"]  # Example relevant resumes

accuracy_metrics = evaluate_accuracy(top_cv_ids, ground_truth, job_vector, preprocessed_cv_data)
print("✅ Accuracy Metrics:", accuracy_metrics)

# Step 8: AI recommendation
prompt = f"""
HR is looking for the best candidate for the following job description: "{job_description}".
Here are the top resumes matched:
{top_cv_files_name}

Please confirm if these candidates match the requirements. If none are relevant, respond with: 'No relevant CVs found.'
"""
final_response = llm.invoke(prompt)

print("✅ AI Recommendation:", final_response)
