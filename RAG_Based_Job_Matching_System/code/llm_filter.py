from store_embeddings import store_embeddings
from preprocess_cv import preprocess_text
from extract_cv import extract_cv_text, get_cv_id_to_filename
from search_cv import search_top_cv
from langchain.llms import cohere

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
vectorstore, cv_ids = store_embeddings(preprocessed_cv_data)  # ✅ Store embeddings of preprocessed text

print("✅ CV embeddings stored successfully!")

# Step 4: Get job description
job_description = input("Enter job description: ")

# Step 5: Retrieve matching CVs
top_cv_ids = search_top_cv(job_description, vectorstore, cv_ids)

print("✅ Top CVs retrieved successfully!")

# Step 6: Get file names for matched CVs
file_name_dict = get_cv_id_to_filename()
top_cv_files_name = [file_name_dict.get(cv_id, "Unknown File") for cv_id in top_cv_ids]

print("✅ Top CV file names:", top_cv_files_name)

# Step 7: AI recommendation
prompt = f"""
HR is looking for the best candidate for the following job description: "{job_description}".
Here are the top resumes matched:
{top_cv_files_name}

Please confirm if these candidates match the requirements. If none are relevant, respond with: 'No relevant CVs found.'
"""
final_response = llm.invoke(prompt)

print("✅ AI Recommendation:", final_response)
