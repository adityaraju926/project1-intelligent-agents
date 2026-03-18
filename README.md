# Reducing Bias in AI Recruitment
## Focal Property

The goal of this project is to reduce gender bias in AI recruitment systems. AI models used for hiring can unintentionally reinforce existing gender biases, favoring one group over another even when candidates are equally qualified. This project aims to reduce this bias by fine-tuning models to focus solely on qualifications to make  hiring decisions regardless of the candidate's gender.


## Dataset 
The dataset used for this project is the [Kaggle resume dataset](https://www.kaggle.com/datasets/dare2dream/resume-extraction) (`data/raw_data.csv`). This dataset contains ~1M rows across 10 columns (Gender, Education, Specialization, Skills, Certifications, Job_status, Job_title, Yearly salary, etc.).

## Exploratory Data Analysis (EDA)
EDA is performed in `eda.ipynb` on the raw data.  

### Key findings:
 **Missing values**: Gender was missing for ~23% of rows (~244K). Several other fields (Job_title, Highest Qualification) had missing values as well  
   
 **Gender distribution**: The dataset skews male, with about a 2:1 male-to-female ratio among non-null entries  
        <img width="537" height="354" alt="Screenshot 2026-03-17 at 2 36 31 PM" src="https://github.com/user-attachments/assets/4c19d10b-f627-47b6-9ed6-c4dbda286148" />
  
 **Employment by gender**: There is a significant difference in employment rates between male and female candidates, with male candidates being offered the job more than female candidates  
        <img width="537" height="409" alt="Screenshot 2026-03-17 at 2 37 40 PM" src="https://github.com/user-attachments/assets/4c48d52f-ef7e-4563-8523-2efcf7bc923e" />

  **Salary by gender**: Average salary differed between genders, with males earning slightly more on average  
        <img width="489" height="415" alt="Screenshot 2026-03-17 at 2 38 48 PM" src="https://github.com/user-attachments/assets/69eb52bc-ceae-42ea-b172-ce9819cdc57d" />

  **Gender distribution by job title**: Across the top 15 job titles, gender representation was uneven, with most roles being dominated by males  
        <img width="788" height="661" alt="Screenshot 2026-03-17 at 2 36 13 PM" src="https://github.com/user-attachments/assets/032c847c-839a-4fde-a3ca-b34b4ba24776" />

A counterfactual dataset, `flipped_dataset.csv` was created by flipping the gender in each row and adding it to the orignial dataset to have a more even gender and role distribution to reduce the gender bias in the dataset.

---

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

---

## Fine-Tuning

Fine-tuning is performed in `finetune.ipynb`

**Model:** [unsloth/llama-2-7b-bnb-4bit](https://huggingface.co/unsloth/llama-2-7b-bnb-4bit)

### Pipeline:

**Data Preparation:** Split data into 80/20 train–validation sets

**Training Strategy:** Applied LoRA (Low-Rank Adaptation) for efficient parameter tuning. Combined with 4-bit quantization to reduce memory usage and Unsloth for faster training

**Training Method:** Used completion-only training and computed loss on response tokens

**Experiments:** Trained on subsets of 5K, 10K, and 20K samples to compare performance

---

## Evaluation

The evaluation covers the base model and the fine-tuned models (5K, 10K, and 20K samples) using a test set of 500 samples from `data/test.csv`. Each candidate sample is formatted into the same instruction-style prompt used during training. The model generates a response, which is then converted into a binary decision: **hired (1)** or **not hired (0)**.


### Metrics
- **Overall Accuracy:** The percentage of predictions that match the ground truth label (`is_employed`). This indicates whether fine-tuning improves or maintains the model’s ability to make correct hiring decisions.

- **Accuracy by Gender:** Accuracy is calculated separately for male and female candidates to check whether the model performs differently across genders.

- **Hire Rate:** The proportion of candidates predicted as “hired” within each gender, regardless of the ground truth. This shows if the model favors one group over the other.

- **Parity Gap:** The absolute difference between male and female hire rates:
`Parity Gap = | Hire Rate (Male) - Hire Rate (Female) |`
A value of 0 means men and women are hired at the same rate. Larger values mean a stronger imbalance between genders.

---

## Results

| Model        | Accuracy | Male Accuracy | Female Accuracy | Hire Rate (Male) | Hire Rate (Female) | Parity Gap |
|-------------|---------|---------------|----------------|-----------------|------------------|------------|
| Base LLaMA-2 | 0.638  | 0.613         | 0.661          | 1.000           | 1.000            | 0.000      |
| 5K Samples  | 0.938   | 0.934         | 0.942          | 0.605           | 0.650            | 0.045      |
| 10K Samples | 0.950   | 0.959         | 0.942          | 0.597           | 0.634            | 0.038      |
| 20K Samples | 0.950   | 0.959         | 0.942          | 0.580           | 0.611            | 0.031      |

---

## Conclusion

These results show that fine-tuning significantly improves model performance, increasing accuracy from 0.638 in the base model to 0.950 with larger training datasets. The gap in hiring rates between male and female candidates decreased as more data was used which suggests that fine-tuning helped reduce bias in the hiring decisions. Overall, the results show that fine-tuning can make AI recruitment systems are more accurate and more fair, though small disparities between groups still remain
