
import streamlit as st
import requests, time, re, json
import PyPDF2
import pandas as pd
from io import BytesIO
from typing import List, Dict
import json
import re 

                           


st.set_page_config(page_title="HR Recruitment Assistant",
                   page_icon="👥", layout="wide")
st.markdown("""
<style>
.main-header {font-size:2.5rem;color:#1f77b4;text-align:center;padding:1rem 0;}
.sub-header  {font-size:1.5rem;color:#ff7f0e;padding:.5rem 0;}
</style>""", unsafe_allow_html=True)


def extract_text_from_pdf(pdf_file) -> str:
    try:
        reader, text = PyPDF2.PdfReader(pdf_file), ""
        for p in reader.pages: text += p.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_contact_info(txt:str)->Dict[str,str]:
    email_pat = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    email  = re.findall(email_pat, txt)
    name   = "Unknown"
    for line in txt.splitlines()[:5]:
        l = line.strip()
        if l and len(l.split())<=4 and not any(c.isdigit() for c in l)\
           and not any(k in l.lower() for k in
                       ['resume','cv','curriculum','phone','email']):
            name = l; break
    return {"name":name, "email":email[0] if email else "Not found"}

import os
import base64
import requests

# Reads from secrets.toml or .env on Streamlit Cloud
OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
OLLAMA_USER = os.getenv("OLLAMA_USER", "")
OLLAMA_PASS = os.getenv("OLLAMA_PASS", "")

def call_ollama(prompt: str, retries: int = 3, **extra_params) -> str:
    headers = {}
    
    # Add Basic Auth if provided
    if OLLAMA_USER and OLLAMA_PASS:
        token = base64.b64encode(f"{OLLAMA_USER}:{OLLAMA_PASS}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"

    payload = {
        "model": "mistral:7b",   # or any model you use
        "prompt": prompt,
        "stream": False
    }
    payload.update(extra_params)

    for _ in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=120)
            if r.status_code == 200:
                return r.json().get("response", "")
            st.error(f"Ollama error {r.status_code}: {r.text}")
            return ""
        except Exception as e:
            st.error(f"Ollama call failed: {e}")
            return ""


def analyze_resume(job_description: str, resume_text: str) -> Dict[str, any]:
    """
    Call the LLM, expect JSON, parse it safely.  Fallback to regex if needed.
    Returns:
        {"score": int,
         "strengths": [str, …],
         "gaps": [str, …],
         "full": raw_response_text}
    """
    prompt = f"""
You are an expert HR recruiter with deep technical and domain expertise across industries.

TASK: Perform a comprehensive analysis comparing the resume to the job description. Evaluate beyond surface-level keyword matching to assess true fit across multiple dimensions.

ANALYSIS FRAMEWORK:
• Technical Alignment: Evaluate tech stack compatibility, tool proficiency, and technical depth
• Domain Expertise: Assess industry knowledge, business context understanding, and relevant problem-solving experience  
• Experience Level: Match seniority, scope of responsibility, and leadership requirements
• Transferable Skills: Identify relevant skills that may not be exact matches but demonstrate adaptability
• Project Complexity: Compare scale, complexity, and impact of past work to role requirements
• Learning Trajectory: Evaluate growth pattern, adaptability, and potential for role evolution
• Methodologies: Assess familiarity with relevant frameworks, processes, and best practices

Compare the resume to the JD and return a JSON object with keys:
  "score"      – integer 0-100
  "strengths"  – list of 3 short bullet points
  "gaps"       – list of missing / weak areas (may be empty)

Return ONLY the JSON.  No markdown fences, no extra text outside the JSON.

Job Description:
{job_description}

Resume:
{resume_text}
"""

    raw = call_ollama(prompt)
    if not raw:                       # network / timeout failure
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
    except (json.JSONDecodeError, TypeError, ValueError):
        
        score_match = re.search(r"SCORE:\s*(\d+)", raw)
        score = int(score_match.group(1)) if score_match else 0

        strengths_match = re.search(
            r"KEY_STRENGTHS:\s*(.*?)(?:GAPS:|$)", raw, re.S)
        strengths_block = strengths_match.group(1) if strengths_match else ""
        strengths = [s.strip("-• \n") for s in strengths_block.splitlines()
                     if s.strip()][:3]

        gaps_match = re.search(r"GAPS:\s*(.*)", raw, re.S)
        gaps_block = gaps_match.group(1) if gaps_match else ""
        gaps = [g.strip("-• \n") for g in gaps_block.splitlines()
                if g.strip()]

        return {
            "score": score,
            "strengths": strengths,
            "gaps": gaps,
            "full": raw,
        }
def generate_questions(job_description: str,
                       resume_text: str,
                       num_questions: int) -> List[Dict[str, str]]:
    """
    Return a list of dictionaries:
    [
      {"question": "...",
       "assesses": "...",
       "good_answer": "..."},
      ...
    ]
    """
    prompt = f"""
    You are an experienced HR interviewer. Generate exactly {num_questions} behavioral interview questions that can be specifically used for screening the candidates:
    • Open-ended (no yes/no questions)
    • Specific to the role and candidate background
    • Designed to assess domain knowledge, and cultural fit
    • Focused on past experiences and specific examples

    Examples - 
    1. How many years of experience do you have in this specific tech stack or doamin?
    2. What kind of bussines impact did you have?

    **FORMAT:** Return ONLY valid JSON – an array where each element
    is an object with keys:
      - "question"
      - "assesses"
      - "good_answer"

    Example (for two questions, spacing unimportant):
    [
      {{"question": "…", "assesses": "…", "good_answer": "…"}},
      {{"question": "…", "assesses": "…", "good_answer": "…"}}
    ]

    Do NOT wrap the JSON in markdown fences and do NOT add commentary.

    -------------------------------
    Job Description  ▶
    {job_description}

    Candidate Resume ▶
    {resume_text}
    """

    raw = call_ollama(prompt)
    if not raw:
        return []

    
    cleaned = raw.strip()
    cleaned = re.sub(r"^```json|```$|```", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        # normalise keys just in case
        return [
            {
                "question": d.get("question", "").strip(),
                "assesses": d.get("assesses", "").strip(),
                "good_answer": d.get("good_answer", "").strip()
            }
            for d in data
            if isinstance(d, dict) and d.get("question")
        ]
    except json.JSONDecodeError:
        # fallback: at least show the raw output in the UI
        st.error("❌ LLM did not return valid JSON. Raw output shown below ⬇️")
        st.code(raw)
        return []

st.markdown('<h1 class="main-header">🎯 HR Recruitment Assistant</h1>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📊 Resume Analyzer", "❓ Interview Questions"])

with tab1:
    jd = st.text_area("📋 Job Description", height=300,
                      placeholder="Paste JD here…")
    pdfs = st.file_uploader("📄 Upload Resumes (PDF)", type=["pdf"],
                            accept_multiple_files=True)
    if st.button("🔍 Analyze"):
        if not jd or not pdfs:
            st.warning("Provide both JD and resumes."); st.stop()
        prog = st.progress(0); results=[]
        for i,pdf in enumerate(pdfs,1):
            txt = extract_text_from_pdf(pdf)
            info = extract_contact_info(txt)
            ana  = analyze_resume(jd, txt)
            results.append({"filename":pdf.name,**info,**ana})
            prog.progress(i/len(pdfs))
        results.sort(key=lambda x:x["score"], reverse=True)
        st.subheader("🏆 Ranking")
        for rk,r in enumerate(results,1):
            with st.expander(f"#{rk} {r['name']} – {r['score']}"):
                st.write(f"**Email:** {r['email']}")
                st.write("**Strengths:**"); st.write("\n".join(f"- {s}" for s in r['strengths'][:3]))
                st.write("**Gaps:**");       st.write("\n".join(f"- {g}" for g in r['gaps'][:3]) or "None")
        st.subheader("📊 Summary")
        st.dataframe(pd.DataFrame([{"Rank":i+1,"Name":r['name'],
                                    "Email":r['email'],"Score":r['score']}
                                   for i,r in enumerate(results)]), use_container_width=True)

with tab2:
    st.markdown('<h2 class="sub-header">Interview Question Generator</h2>',
                unsafe_allow_html=True)

    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Job Description")
        jd_interview = st.text_area(
            label="Paste the job description",
            height=250,
            key="jd_interview",
            placeholder="Enter the job requirements…"
        )

    with col2:
        st.markdown("### 📄 Upload Resume")
        resume_file = st.file_uploader(
            label="Choose a PDF file",
            type=["pdf"],
            key="resume_interview"
        )

        num_questions = st.slider(
            label="Number of questions to generate",
            min_value=3,
            max_value=15,
            value=5
        )

    
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
                st.error("Failed to generate questions. Please try again.")
                st.stop()

            
            st.markdown("### 📝 Generated Interview Questions")
            for i, q in enumerate(questions, 1):
                st.markdown(f"**Question {i}:** {q['question']}")
