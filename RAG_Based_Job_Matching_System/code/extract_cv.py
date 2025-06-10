import os
import uuid
import PyPDF2
import docx



#cv_folder = os.path.abspath("RAG_Based_Job_Matching_System/Resume")
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
cv_folder = os.path.join(base_dir, "..", "Resume")  # Move up from `code/` to `Resume/`

cv_data = {}
cv_id_to_filename = {}  # Dictionary to store CV ID → File Name mapping

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text.strip()

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return " ".join([para.text for para in doc.paragraphs])

def extract_cv_text():
    global cv_id_to_filename  # Ensure mapping persists

    for filename in os.listdir(cv_folder):
        file_path = os.path.join(cv_folder, filename)
        
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            continue
        
        unique_id = "cv_" + str(uuid.uuid4())[:8]
        cv_data[unique_id] = text
        cv_id_to_filename[unique_id] = filename  # Store file name mapping

    return cv_data

def get_cv_id_to_filename():
    """Returns a dictionary mapping CV IDs to their file names."""
    return cv_id_to_filename
