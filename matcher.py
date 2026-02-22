from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# Load BERT model 
@lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

#Skill List
skills_list = [
    "python",
    "machine learning",
    "deep learning",
    "nlp",
    "tensorflow",
    "pytorch",
    "scikit learn",
    "pandas",
    "numpy",
    "sql",
    "aws",
    "docker",
    "fastapi",
    "flask",
    "bert",
    "transformers"
]

# BERT Similarity
def bert_similarity(resumes, job_desc):
    resume_embeddings = model.encode(resumes)
    job_embedding = model.encode([job_desc])
    scores = cosine_similarity(job_embedding, resume_embeddings)[0]
    return scores

# Skill Matching
def skill_score(resume_text, job_text):
    resume_text = resume_text.lower()
    job_text = job_text.lower()

    job_skills = [s for s in skills_list if s in job_text]

    if not job_skills:
        return 0, []

    matched = [s for s in job_skills if s in resume_text]

    score = len(matched) / len(job_skills)
    return score, matched

# Final Score
def match_resumes(resumes, job_desc):
    bert_scores = bert_similarity(resumes, job_desc)
    results = []

    for i, resume in enumerate(resumes):
        s_score, matched_skills = skill_score(resume, job_desc)

        final_score = (
            0.7 * bert_scores[i] +
            0.3 * s_score
        )

        results.append({
            "score": final_score,
            "match_percent": round(final_score * 100, 2),
            "bert": round(bert_scores[i], 2),
            "skill_score": round(s_score, 2),
            "matched_skills": matched_skills
        })

    return results