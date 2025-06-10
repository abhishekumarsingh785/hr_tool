import streamlit as st
import requests, time, re, json, os, base64
#import PyPDF2
import pandas as pd
from io import BytesIO
from typing import List, Dict
import pdfplumber
#from dotenv import load_dotenv

# Load environment variables (for local dev)
#load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="HR Recruitment Assistant", page_icon="👥", layout="wide")

st.markdown("""
<style>
.main-header {font-size:2.5rem;color:#1f77b4;text-align:center;padding:1rem 0;}
.sub-header  {font-size:1.5rem;color:#ff7f0e;padding:.5rem 0;}
</style>
""", unsafe_allow_html=True)

# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_file) -> str:
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def extract_contact_info(txt: str) -> Dict[str, str]:
    email_pat = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    email = re.findall(email_pat, txt)
    name = "Unknown"
    for line in txt.splitlines()[:5]:
        l = line.strip()
        if l and len(l.split()) <= 4 and not any(c.isdigit() for c in l) \
                and not any(k in l.lower() for k in ['resume', 'cv', 'curriculum', 'phone', 'email']):
            name = l
            break
    return {"name": name, "email": email[0] if email else "Not found"}

# --- GROQ LLM Call ---
GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_groq(prompt: str, retries: int = 3, model: str = "llama3-70b-8192", temperature: float = 0.2) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "stream": False
    }

    for _ in range(retries):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
            else:
                st.error(f"Groq API error {r.status_code}: {r.text}")
                return ""
        except Exception as e:
            st.error(f"Groq API call failed: {e}")
            return ""

# --- Resume Analysis ---
def analyze_resume(job_description: str, resume_text: str) -> Dict[str, any]:
    prompt = f"""
You are an expert HR recruiter with deep technical and domain expertise.

TASK: Compare the resume to the job description. Go beyond keywords to assess:

- Technical Alignment
- Domain Expertise
- Experience Level
- Transferable Skills
- Project Complexity
- Learning Trajectory
- Methodologies

Return ONLY JSON with:
  "score": int (0-100),
  "strengths": list of 3 bullets,
  "gaps": list (may be empty)

Job Description:
{job_description}

Resume:
{resume_text}
"""
    raw = call_groq(prompt)
    if not raw:
        return {"score": 0, "strengths": [], "gaps": [], "full": ""}

    cleaned = re.sub(r"^```json|```$|```", "", raw.strip()).strip()

    try:
        data = json.loads(cleaned)
        return {
            "score": int(data.get("score", 0)),
            "strengths": data.get("strengths", [])[:3],
            "gaps": data.get("gaps", []),
            "full": raw,
        }
    except Exception:
        return {"score": 0, "strengths": [], "gaps": [], "full": raw}

# --- Interview Questions ---
def generate_questions(job_description: str, resume_text: str, num_questions: int) -> List[Dict[str, str]]:
    prompt = f"""
You are an experienced HR interviewer. Generate exactly {num_questions} behavioral interview questions.

Must be:
- Open-ended
- Based on resume & JD
- Assess domain fit, communication, culture

FORMAT:
[
  {{"question": "...", "assesses": "...", "good_answer": "..." }},
  ...
]

Job Description:
{job_description}

Resume:
{resume_text}
"""
    raw = call_groq(prompt)
    if not raw:
        return []

    cleaned = re.sub(r"^```json|```$|```", "", raw.strip()).strip()

    try:
        data = json.loads(cleaned)
        return [
            {
                "question": d.get("question", "").strip(),
                "assesses": d.get("assesses", "").strip(),
                "good_answer": d.get("good_answer", "").strip()
            }
            for d in data if isinstance(d, dict)
        ]
    except Exception:
        st.error("❌ Invalid JSON returned from Groq.")
        st.code(raw)
        return []

# --------------------- UI ---------------------
st.markdown('<h1 class="main-header">🎯 HR Recruitment Assistant</h1>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📊 Resume Analyzer", "❓ Interview Questions"])

# ---- Tab 1 ----
with tab1:
    jd = st.text_area("📋 Job Description", height=300, placeholder="Paste JD here…")
    pdfs = st.file_uploader("📄 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

    if st.button("🔍 Analyze"):
        if not jd or not pdfs:
            st.warning("Provide both JD and resumes.")
            st.stop()

        prog = st.progress(0)
        results = []

        for i, pdf in enumerate(pdfs, 1):
            txt = extract_text_from_pdf(pdf)
            info = extract_contact_info(txt)
            ana = analyze_resume(jd, txt)
            results.append({"filename": pdf.name, **info, **ana})
            prog.progress(i / len(pdfs))

        results.sort(key=lambda x: x["score"], reverse=True)

        st.subheader("🏆 Ranking")
        for rk, r in enumerate(results, 1):
            with st.expander(f"#{rk} {r['name']} – {r['score']}"):
                st.write(f"**Email:** {r['email']}")
                st.write("**Strengths:**")
                st.write("\n".join(f"- {s}" for s in r['strengths']))
                st.write("**Gaps:**")
                st.write("\n".join(f"- {g}" for g in r['gaps']) or "None")

        st.subheader("📊 Summary")
        df = pd.DataFrame([{"Rank": i+1, "Name": r['name'], "Email": r['email'], "Score": r['score']} for i, r in enumerate(results)])
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download Rankings", df.to_csv(index=False), "rankings.csv")

# ---- Tab 2 ----
with tab2:
    st.markdown('<h2 class="sub-header">Interview Question Generator</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Job Description")
        jd_interview = st.text_area("Paste the job description", height=250, key="jd_interview")

    with col2:
        st.markdown("### 📄 Upload Resume")
        resume_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="resume_interview")
        num_questions = st.slider("Number of questions to generate", 3, 15, 5)

    if st.button("🎯 Generate Interview Questions", type="primary"):
        if not (jd_interview and resume_file):
            st.warning("Please provide both a job description and a resume.")
            st.stop()

        with st.spinner("Generating interview questions…"):
            resume_text = extract_text_from_pdf(resume_file)
            if not resume_text:
                st.error("Could not extract text from the PDF.")
                st.stop()

            questions = generate_questions(jd_interview, resume_text, num_questions)
            if not questions:
                st.error("No questions generated.")
                st.stop()

            st.markdown("### 📝 Generated Interview Questions")
            for i, q in enumerate(questions, 1):
                with st.expander(f"Question {i}: {q['question']}"):
                    st.markdown(f"**Assesses:** {q['assesses']}")
                    st.markdown(f"**Good Answer:** {q['good_answer']}")
