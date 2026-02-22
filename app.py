import streamlit as st
import numpy as np
from parser import extract_text_from_pdf
from preprocess import preprocess_text
from matcher import match_resumes


st.set_page_config(
    page_title="AI ATS Resume Screener",
    layout="wide",
    page_icon="📄"
)

st.title("📄 AI-Powered Resume Screening System")
st.markdown("Semantic resume matching using BERT + ATS-style skill scoring")

st.sidebar.header("Settings")

top_k = st.sidebar.slider(
    "Number of top candidates to display",
    min_value=1,
    max_value=20,
    value=5
)

show_details = st.sidebar.checkbox("Show detailed scores", value=True)

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

job_desc = st.text_area(
    "Paste Job Description",
    height=200
)

analyze = st.button("Analyze Candidates")

def match_level(score):
    if score >= 80:
        return "🟢 Excellent Match"
    elif score >= 60:
        return "🟡 Good Match"
    elif score >= 40:
        return "🟠 Average Match"
    else:
        return "🔴 Low Match"


if analyze:
    if uploaded_files and job_desc:

        with st.spinner("Processing resumes..."):

            resumes = []
            file_names = []

            progress = st.progress(0)

            for i, file in enumerate(uploaded_files):
                text = extract_text_from_pdf(file)
                text = preprocess_text(text)
                resumes.append(text)
                file_names.append(file.name)

                progress.progress((i + 1) / len(uploaded_files))

            job_desc_clean = preprocess_text(job_desc)

            results = match_resumes(resumes, job_desc_clean)

        sorted_indices = np.argsort([r["score"] for r in results])[::-1]

        st.success("Analysis Complete!")


        best_idx = sorted_indices[0]
        best_result = results[best_idx]

        st.subheader("🏆 Best Candidate")
        st.metric(
            label=file_names[best_idx],
            value=f"{best_result['match_percent']}%"
        )
        st.write(match_level(best_result["match_percent"]))

        st.divider()

        st.subheader("Candidate Rankings")

        display_count = min(top_k, len(sorted_indices))

        for rank in range(display_count):
            idx = sorted_indices[rank]
            r = results[idx]

            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"### {rank+1}. {file_names[idx]}")
                    st.write(match_level(r["match_percent"]))

                with col2:
                    st.metric("Match %", f"{r['match_percent']}%")

                if show_details:
                    d1, d2 = st.columns(2)
                    d1.write(f"**Semantic Score:** {r['bert']}")
                    d2.write(f"**Skill Score:** {r['skill_score']}")

                st.write("**Matched Skills:**")
                if r["matched_skills"]:
                    st.write(", ".join(r["matched_skills"]))
                else:
                    st.write("No key skills matched")

                st.divider()

    else:
        st.warning("Please upload resumes and enter a job description.")