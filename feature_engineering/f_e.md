# Feature Engineering for Gender Bias in Recruitment

This notebook prepares resume data for downstream modeling aimed at reducing gender bias in recruitment screening. It loads raw resume extraction data, preprocesses it, creates job-relevant features, and outputs a clean feature matrix ready for modeling.

## Pipeline Overview

### 1. Data Loading
- Loads `data/resume_extraction.csv`
- Handles malformed rows (`on_bad_lines="skip"`), encoding issues, and mixed types
- Strips column names (e.g., trailing space in `skills`)

### 2. Initial Exploration
- Displays shape, columns, data types, and missing value counts
- Shows value distributions for Gender, Education, and Job_status

### 3. Preprocessing
- **Missing values:** Maps placeholders ("No", "NA", "NO", "") to NaN for text columns
- **Salary:** Coerces to numeric; invalid values become NaN
- **Text normalization:** Standardizes Education (e.g., "b.e" → "BE"), Specialization (lowercase), and Gender (strip whitespace)

### 4. Skills and Interests Parsing
- Parses semicolon- and comma-separated skills and interests into lists
- Deduplicates and strips whitespace
- Creates `skills_list` and `interests_list` columns

### 5. Feature Creation
Builds job-relevant features from the raw and parsed data (see below).

### 6. Gender Bias Considerations
- Assembles final feature matrix with job-relevant features only
- Keeps Gender as a protected attribute for bias evaluation (excluded from model input)
- Fills missing salary with median; missing categorical values with "unknown" or "Other"

### 7. Output and Validation
- Saves to `data/resume_features.csv`
- Validates: duplicate count, row count, categorical distributions, numeric summaries

---

## New Features Created

| Feature | Type | Description |
|---------|------|-------------|
| **education_level** | Ordinal (1–3) | Diploma=1, Bachelors=2, Masters=3 |
| **specialization_domain** | Categorical | STEM, Business, Humanities, or Other |
| **has_certification** | Binary | 1 if has certification, 0 otherwise |
| **is_employed** | Binary | 1 if employed (Job_status=Yes), 0 otherwise |
| **skill_count** | Numeric | Number of distinct skills per resume |
| **interest_count** | Numeric | Number of interests per resume |
| **has_programming_skills** | Binary | 1 if any programming skill (Python, Java, SQL, etc.), 0 otherwise |
| **has_soft_skills** | Binary | 1 if any soft skill (Communication, Leadership, etc.), 0 otherwise |
| **tech_skill_count** | Numeric | Count of programming-related skills |
| **soft_skill_count** | Numeric | Count of soft skills |
| **education_job_match** | Binary | 1 if specialization aligns with job title, 0 otherwise |
| **highest_qualification_level** | Ordinal (1–3) | Bachelor=1, Master=2, PhD=3 |
| **yearly_salary** | Numeric | Yearly salary in pounds |
| **salary_bucket** | Categorical | low (<50k), medium (50–70k), high (>70k), or unknown |
| **job_title_length** | Numeric | Character count of job title |
| **job_title_clean** | Text | Normalized job title (lowercase, stripped) for clustering/embedding |
