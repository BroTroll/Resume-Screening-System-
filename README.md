```markdown
# AI Resume Screening System (ATS)

An AI-powered Resume Screening System that automatically ranks resumes based on their relevance to a job description using **Natural Language Processing (NLP)** and **BERT embeddings**.

This project simulates the core functionality of an **Applicant Tracking System (ATS)** used by recruiters.

---

## Features

- Upload multiple resumes (PDF)
- Extract and preprocess resume text
- Semantic matching using **BERT (Sentence Transformers)**
- Skill-based matching for Machine Learning Engineer roles
- ATS-style scoring:
  - 70% Semantic similarity
  - 30% Skill match
- Candidate ranking based on match percentage
- Interactive **Streamlit web interface**
- Displays:
  - Match %
  - Semantic score
  - Skill score
  - Matched skills
  - Top candidate highlight

---

## System Workflow

```

PDF Resume
↓
Text Extraction (pdfplumber)
↓
Text Preprocessing (NLTK)
↓
BERT Embedding (Sentence Transformers)
↓
Skill Matching
↓
Final ATS Score
↓
Candidate Ranking (Streamlit UI)

````

---

## Tech Stack

- Python
- Streamlit
- Sentence Transformers (BERT)
- Scikit-learn
- NLTK
- pdfplumber
- NumPy

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/resume-screening-system.git
cd resume-screening-system
````

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Download NLTK resources

Run once:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Run the Application

```bash
streamlit run app.py
```

The app will open in your browser.

---

## Usage

1. Upload multiple resume PDFs
2. Paste a job description
3. Click **Analyze Candidates**
4. View ranked candidates with match percentage and skill details

---

## Example Use Case

Designed for:

* Machine Learning Engineer hiring
* Campus placement automation
* Resume filtering for recruiters
* AI-based HR tools

---

## Project Structure

```
resume-screening-system/
│
├── app.py
├── parser.py
├── preprocess.py
├── matcher.py
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Support for multiple job roles
* Resume–job gap analysis
* Experience extraction
* Export results to CSV/Excel
* Deploy on cloud

---



## License

This project is for educational and portfolio purposes.


## Images/Dashboard
<img width="1901" height="985" alt="Screenshot 2026-02-22 173006" src="https://github.com/user-attachments/assets/9fd2456a-4586-4982-a641-cdd4e86ef441" />


<img width="1878" height="899" alt="Screenshot 2026-02-22 173054" src="https://github.com/user-attachments/assets/89e10f6f-a435-459e-98f8-82272370f45f" />

```




