import os
import fitz  # PyMuPDF
import csv
import datetime
import requests
import time
import streamlit as st
import pandas as pd
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time as timer
import docx
from st_aggrid import AgGrid, GridOptionsBuilder

# ---------------------------
# OLLAMA SETTINGS
# ---------------------------
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"

# ---------------------------
# EXTRACT TEXT FROM FILE
# ---------------------------
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading {file_path}: {e}"

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error reading {file_path}: {e}"

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return "Unsupported file format."

# ---------------------------
# EXTRACT CONTACT INFO
# ---------------------------
def extract_contact_info(text):
    name_match = re.search(r"(?i)(?:name[:\s]*)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
    phone_match = re.search(r"(\+?\d[\d\s\-\(\)]{7,})", text)
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)

    name = name_match.group(1) if name_match else "Not Found"
    phone = phone_match.group(1) if phone_match else "Not Found"
    email = email_match.group(0) if email_match else "Not Found"

    return name.strip(), phone.strip(), email.strip()

# ---------------------------
# SCORE CV AGAINST JOB DESCRIPTION (OLLAMA)
# ---------------------------
def score_cv_with_ollama(cv_text, job_description, retries=3):
    prompt = f"""
    You are an AI recruiter evaluating a candidate's resume against a job description.

    Provide:
    1. A match score from 0 to 100, based on the following general criteria:
       - Skills Match: Does the candidate have the key technical, professional, or domain-specific skills required for the job?
       - Relevant Experience: How well do the candidate's previous job roles, projects, and accomplishments align with the job requirements?
       - Education & Certifications: Is the candidate's educational background or certifications relevant to the job?
       - Soft Skills & Communication: How well does the candidate demonstrate teamwork, problem-solving, communication, and leadership abilities, if relevant?
       - Job-Specific Requirements: Any additional specific qualifications, experience, or traits mentioned in the job description that the candidate fulfills (e.g., specific technologies, languages, or domain expertise).

    2. A brief explanation (40–50 words) explaining why the candidate was or was not a good fit for the position.

    Job Description: {job_description}

    Candidate CV: {cv_text}

    Output format (strictly follow this format):
    Score: <numeric score>
    Reason: <40–50 word explanation>
    """
    
    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
            )
            response.raise_for_status()
            return response.json().get("response", "No response.")
        except Exception as e:
            time.sleep(5 + attempt * 5)
            if attempt == retries - 1:
                return f"Error: {e}"

# ---------------------------
# PARSE RESULTS AND SHORTLIST
# ---------------------------
def parse_and_shortlist(results, threshold=70):
    parsed = []
    for file, result in results:
        try:
            score_match = re.search(r"Score:\s*(\d+)", result)
            reason_match = re.search(r"Reason:\s*(.+)", result, re.DOTALL)

            score = int(score_match.group(1)) if score_match else 0
            reason = reason_match.group(1).strip() if reason_match else "No reason provided."
            shortlisted = "Yes" if score >= threshold else "No"
        except Exception as e:
            score = 0
            shortlisted = "No"
            reason = f"Parsing error: {e}"
        parsed.append({
            "Filename": file,
            "Score": score,
            "Shortlisted": shortlisted,
            "Reason": reason
        })
    return parsed

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.set_page_config(page_title="ResuAI", layout="wide")

st.markdown("""<h1 style="text-align: center; font-size: 40px; color: #4CAF50;">ResuAI</h1>""", unsafe_allow_html=True)

st.markdown("Upload CVs and a job description. This app scores and shortlists candidates.")

job_description = st.text_area("📋 Paste Job Description Here", height=200)

uploaded_files = st.file_uploader("📁 Upload CVs (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

threshold = st.sidebar.slider("🔧 Scoring Threshold", 0, 100, 70, 1)

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

if st.button("🔍 Run Shortlisting"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one CV.")
    else:
        start_time = timer.time()
        with st.spinner("Scoring CVs..."):
            results = []
            contact_info = []
            with TemporaryDirectory() as tmpdir:
                for file in uploaded_files:
                    file_path = os.path.join(tmpdir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.read())

                def process_cv(file_name, file_path, job_desc):
                    cv_text = extract_text(file_path)
                    result = score_cv_with_ollama(cv_text[:12000], job_desc)
                    return (file_name, result, cv_text)

                progress_bar = st.progress(0)
                status_text = st.empty()
                total = len(uploaded_files)

                with ThreadPoolExecutor() as executor:
                    futures = []
                    for file in uploaded_files:
                        file_path = os.path.join(tmpdir, file.name)
                        futures.append(executor.submit(process_cv, file.name, file_path, job_description))

                    for i, future in enumerate(as_completed(futures)):
                        file_name, result, cv_text = future.result()
                        results.append((file_name, result))
                        progress_bar.progress((i + 1) / total)
                        status_text.text(f"Processed {i + 1} of {total} CVs")

                data = parse_and_shortlist(results, threshold)
                df = pd.DataFrame(data)
                st.session_state.results_df = df

                # Save contact info for shortlisted candidates
                for row in data:
                    if row["Shortlisted"] == "Yes":
                        file_path = os.path.join(tmpdir, row["Filename"])
                        text = extract_text(file_path)
                        name, phone, email = extract_contact_info(text)
                        contact_info.append({
                            "Filename": row["Filename"],
                            "Name": name,
                            "Phone": phone,
                            "Email": email
                        })

                # Create contact CSV
                contact_df = pd.DataFrame(contact_info)
                st.session_state.contact_csv_data = contact_df.to_csv(index=False).encode()

                # Create main CSV
                csv_bytes = df.to_csv(index=False).encode()
                st.session_state.csv_data = csv_bytes

                elapsed = timer.time() - start_time
                st.success(f"✅ Scoring complete in {elapsed:.2f} seconds!")

# ---------------------------
# DISPLAY RESULTS
# ---------------------------
if not st.session_state.results_df.empty:
    st.subheader("📄 Results Table")
    gb = GridOptionsBuilder.from_dataframe(st.session_state.results_df)
    gb.configure_pagination()
    gb.configure_default_column(sortable=True, filter=True)
    grid_options = gb.build()
    AgGrid(st.session_state.results_df, gridOptions=grid_options, enable_enterprise_modules=True, height=400)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("📅 Download All Results (CSV)", st.session_state.csv_data, file_name=f"cv_results_{timestamp}.csv", mime="text/csv")
    st.download_button("📅 Download Shortlisted Contacts (CSV)", st.session_state.contact_csv_data, file_name=f"shortlisted_contacts_{timestamp}.csv", mime="text/csv")
