import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load BERT model (once)
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# General Skill List
# ----------------------------
GENERAL_SKILLS = [

    # Programming
    "python", "java", "c++", "c", "javascript", "typescript",
    "go", "ruby", "php", "swift", "kotlin", "r", "matlab",

    # Web
    "html", "css", "react", "angular", "vue", "node.js",
    "express", "django", "flask", "spring boot", "rest api",

    # Database
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle",

    # Data
    "excel", "power bi", "tableau", "data analysis",
    "pandas", "numpy", "statistics",

    # AI / ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "xgboost",

    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes",
    "ci/cd", "jenkins", "terraform",

    # Tools
    "git", "github", "gitlab", "linux", "bash", "jira",

    # Core CS
    "data structures", "algorithms", "oop",
    "system design", "microservices", "api",

    # Testing
    "unit testing", "selenium", "pytest",

    # Soft Skills
    "communication", "teamwork", "leadership",
    "problem solving", "agile", "scrum"
]

# ----------------------------
# Extract required skills from Job Description
# ----------------------------
def extract_required_skills(job_desc):
    job_desc = job_desc.lower()
    return [skill for skill in GENERAL_SKILLS if skill in job_desc]


# ----------------------------
# Skill Matching
# ----------------------------
def skill_match_score(resume_text, required_skills):
    resume_text = resume_text.lower()

    matched = [skill for skill in required_skills if skill in resume_text]

    if len(required_skills) == 0:
        return 0, []

    score = len(matched) / len(required_skills)
    return score, matched


# ----------------------------
# Main Matching Function
# ----------------------------
def match_resumes(resumes, job_desc):

    # BERT embeddings
    job_embedding = model.encode([job_desc])
    resume_embeddings = model.encode(resumes)

    similarities = cosine_similarity(resume_embeddings, job_embedding).flatten()

    # Extract required skills dynamically
    required_skills = extract_required_skills(job_desc)

    results = []

    for i, resume in enumerate(resumes):

        bert_score = float(similarities[i])

        skill_score, matched_skills = skill_match_score(resume, required_skills)

        # ATS Final Score
        final_score = (0.7 * bert_score) + (0.3 * skill_score)

        results.append({
            "bert": round(bert_score, 3),
            "skill_score": round(skill_score, 3),
            "score": round(final_score, 3),
            "match_percent": int(final_score * 100),
            "matched_skills": matched_skills,
            "required_skills": required_skills
        })

    return results
