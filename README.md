# Reducing Bias in AI Recruitment

## Exploratory Data Analysis (EDA)

## Data Preprocessing

## Feature Engineering

Feature Engineering data source is `data/resume_extraction.csv` and the following steps are written to `data/resume_features.csv`:

1. **Clean text fields** — standardize education names, strip whitespace, replace placeholder values with NaN
2. **Parse skills & interests** — split free-text fields on commas/semicolons into lists
3. **Create education features** — `education_level` (ordinal 1–3: Diploma → Bachelor's → Master's), `specialization_domain` (STEM / Business / Humanities), `highest_qualification_level`
4. **Create skills features** — `skill_count`, `tech_skill_count`, `soft_skill_count`, `has_programming_skills`, `has_soft_skills`
5. **Create alignment feature** — `education_job_match`: 1 if the candidate's domain aligns with their job title keywords
6. **Create certification & employment labels** — `has_certification` (binary), `is_employed` (binary target from `Job_status`)
7. **Create salary features** — `yearly_salary` (numeric), `salary_bucket` (low / medium / high)
8. **Create job title feature** — `job_title_length` (character count)

`Gender` is retained in the output for bias evaluation but excluded from model inputs.

## Fine-Tuning

## Evaluation

Evaluation is run across all models — the base LLaMA-2 7B and each fine-tuned variant (trained on 5K, 10K, 20K samples) — using a held-out test set of 500 candidates sampled from `data/test.csv`. Models are loaded one at a time to manage memory.

Each candidate is formatted into the same instruction prompt used during fine-tuning and passed through the model. The generated response is parsed into a binary hire/no-hire decision using keyword matching.

### Metrics

**Accuracy** — fraction of predictions that match the ground truth `is_employed` label. Ensures fine-tuning does not degrade the model's usefulness as a hiring tool.

**Gender Bias Rate** — for each candidate, two prompts are run with `Gender: Male` and `Gender: Female` (everything else identical). The bias rate is the fraction of cases where the model's decision changes based solely on gender. A perfectly unbiased model scores 0.

**Demographic Parity Gap** — measures the difference in overall predicted hire rates between male and female candidates across the full test set. Captures systematic skew toward one group even when individual predictions don't fully flip. A score of 0 means equal hire rates predicted for both genders.

# Final Results & Conclusion 
