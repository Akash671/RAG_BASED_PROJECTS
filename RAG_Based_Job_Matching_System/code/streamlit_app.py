import streamlit as st
from store_embeddings import store_embeddings
from extract_cv import extract_cv_text, get_cv_id_to_filename
from search_cv import search_top_cv
from preprocess_cv import preprocess_text
from langchain.llms import cohere

# Initialize Cohere LLM
COHERE_API_KEY = "your_api_key"  # Create your API key on cohere website and then enter here
llm = cohere.Cohere(cohere_api_key=COHERE_API_KEY, temperature=0.1)

# Streamlit App
st.title("🔎 RAG-Based Job Matching System")

# Step 1: Extract CV Data
if st.button("Extract CVs"):
    st.session_state["cv_data"] = extract_cv_text()
    if not st.session_state["cv_data"]:
        st.error("❌ Error: CV text extraction failed!")
    else:
        st.success("✅ CVs extracted successfully!")

# Step 2: Preprocess CV Data
if st.button("Preprocess CVs"):
    st.session_state["preprocessed_cv_data"] = {
        cv_id: preprocess_text(text) for cv_id, text in st.session_state.get("cv_data", {}).items()
    }
    st.success("✅ CVs preprocessed successfully!")

# Step 3: Store CV Embeddings
if st.button("Store Embeddings"):
    try:
        st.session_state["vectorstore"], st.session_state["cv_ids"] = store_embeddings(st.session_state.get("preprocessed_cv_data", {}))
        st.success("✅ CV embeddings stored successfully!")
    except Exception as e:
        st.error(f"❌ Error storing embeddings: {e}")

# Step 4: Enter Job Description
st.header("Enter the Job Description")
job_description = st.text_area("Enter the job description:")

# Step 5: Search Top CVs
if st.button("Search Top CVs"):
    try:
        top_cv_ids = search_top_cv(
            job_description,
            st.session_state.get("vectorstore"),
            st.session_state.get("cv_ids"),
        )
        file_name_dict = get_cv_id_to_filename()
        st.session_state["top_cv_files_name"] = [file_name_dict.get(cv_id, "Unknown File") for cv_id in top_cv_ids]
        st.write("### Top Matching Resumes:")
        st.write("\n".join(st.session_state["top_cv_files_name"]))
    except Exception as e:
        st.error(f"❌ Error searching CVs: {e}")

# Step 6: Get AI Recommendation
if st.button("Get AI Recommendation"):
    try:
        if "top_cv_files_name" not in st.session_state:
            st.error("❌ Error: Search Top CVs first!")
        else:
            prompt = f"""
            HR is looking for the best candidate for the following job description: "{job_description}".
            Here are the top 5 resumes matched:
            {st.session_state["top_cv_files_name"]}

            Please confirm if these candidates match the requirements. If none are relevant, respond with: 'No relevant CVs found.'
            """
            final_response = llm.invoke(prompt)
            st.write("### AI Recommendation:")
            st.write(final_response)
    except Exception as e:
        st.error(f"❌ Error getting AI recommendation: {e}")
