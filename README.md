# Reducing Bias in AI Recruitment

## Exploratory Data Analysis (EDA)
EDA is performed in `eda.ipynb` on the raw [Kaggle resume dataset](https://www.kaggle.com/datasets/dare2dream/resume-extraction) (`data/raw_data.csv`), which contains ~1M rows across 10 columns (Gender, Education, Specialization, Skills, Certifications, Job_status, Job_title, Yearly salary, etc.).

**Key findings:**
- **Missing values**: Gender was missing for ~23% of rows (~244K). Several other fields (Job_title, Highest Qualification) had missing values as well
- **Gender distribution**: The dataset skews male, with about a 2:1 male-to-female ratio among non-null entries  
        <img width="537" height="354" alt="Screenshot 2026-03-17 at 2 36 31 PM" src="https://github.com/user-attachments/assets/4c19d10b-f627-47b6-9ed6-c4dbda286148" />
  
- **Employment by gender**: There is a significant difference in employment rates between male and female candidates, with male candidates being offered the job more than female candidates  
        <img width="537" height="409" alt="Screenshot 2026-03-17 at 2 37 40 PM" src="https://github.com/user-attachments/assets/4c48d52f-ef7e-4563-8523-2efcf7bc923e" />

- **Salary by gender**: Average salary differed between genders, with males earning slightly more on average  
        <img width="489" height="415" alt="Screenshot 2026-03-17 at 2 38 48 PM" src="https://github.com/user-attachments/assets/69eb52bc-ceae-42ea-b172-ce9819cdc57d" />


- **Gender distribution by job title**: Across the top 15 job titles, gender representation was uneven, with most roles being dominated by males  
        <img width="788" height="661" alt="Screenshot 2026-03-17 at 2 36 13 PM" src="https://github.com/user-attachments/assets/032c847c-839a-4fde-a3ca-b34b4ba24776" />

A counterfactual dataset, `flipped_dataset.csv` was created by flipping the gender in each row and adding it to the orignial dataset to have a more even gender and role distribution to reduce the gender bias in the dataset.

## Feature Engineering

Feature Engineering data source is `data/resume_extraction.csv` and the following steps are written to `data/resume_features.csv`:

1. **Clean text fields**: standardize education names, strip whitespace, replace placeholder values with NaN
2. **Parse skills & interests**: split free-text fields on commas/semicolons into lists
3. **Create education features**: `education_level` (ordinal 1–3: Diploma → Bachelor's → Master's), `specialization_domain` (STEM / Business / Humanities), `highest_qualification_level`
4. **Create skills features**: `skill_count`, `tech_skill_count`, `soft_skill_count`, `has_programming_skills`, `has_soft_skills`
5. **Create alignment feature**: `education_job_match`: 1 if the candidate's domain aligns with their job title keywords
6. **Create certification & employment labels**: `has_certification` (binary), `is_employed` (binary target from `Job_status`)
7. **Create salary features**: `yearly_salary` (numeric), `salary_bucket` (low / medium / high)
8. **Create job title feature**: `job_title_length` (character count)

`Gender` is retained in the output for bias evaluation but excluded from model inputs.

## Fine-Tuning

Fine-tuning is performed in `finetune.ipynb`

**Model:** [unsloth/llama-2-7b-bnb-4bit](https://huggingface.co/unsloth/llama-2-7b-bnb-4bit)

### Pipeline:

**Data Preparation:** Split data into 80/20 train–validation sets

**Training Strategy:** Applied LoRA (Low-Rank Adaptation) for efficient parameter tuning. Combined with 4-bit quantization to reduce memory usage and Unsloth for faster training

**Training Method:** Used completion-only training and computed loss on response tokens

**Experiments:** Trained on subsets of 5K, 10K, and 20K samples to compare performance

## Evaluation

Evaluation is run across all models — the base LLaMA-2 7B and each fine-tuned variant (trained on 5K, 10K, 20K samples) — using a held-out test set of 500 candidates sampled from `data/test.csv`. Models are loaded one at a time to manage memory.

Each candidate is formatted into the same instruction prompt used during fine-tuning and passed through the model. The generated response is parsed into a binary hire/no-hire decision using keyword matching.

### Metrics

**Accuracy** — fraction of predictions that match the ground truth `is_employed` label. Ensures fine-tuning does not degrade the model's usefulness as a hiring tool.

**Gender Bias Rate** — for each candidate, two prompts are run with `Gender: Male` and `Gender: Female` (everything else identical). The bias rate is the fraction of cases where the model's decision changes based solely on gender. A perfectly unbiased model scores 0.

**Demographic Parity Gap** — measures the difference in overall predicted hire rates between male and female candidates across the full test set. Captures systematic skew toward one group even when individual predictions don't fully flip. A score of 0 means equal hire rates predicted for both genders.

# Final Results & Conclusion 
