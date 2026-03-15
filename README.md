# Reducing Bias in AI Recruitment

## Exploratory Data Analysis (EDA)

## Data Preprocessing

## Feature Engineering

## Fine-Tuning

## Evaluation

Evaluation is run across all models — the base LLaMA-2 7B and each fine-tuned variant (trained on 5K, 10K, 20K samples) — using a held-out test set of 500 candidates sampled from `data/test.csv`. Models are loaded one at a time to manage memory.

Each candidate is formatted into the same instruction prompt used during fine-tuning and passed through the model. The generated response is parsed into a binary hire/no-hire decision using keyword matching.

### Metrics

**Accuracy** — fraction of predictions that match the ground truth `is_employed` label. Ensures fine-tuning does not degrade the model's usefulness as a hiring tool.

**Gender Bias Rate** — for each candidate, two prompts are run with `Gender: Male` and `Gender: Female` (everything else identical). The bias rate is the fraction of cases where the model's decision changes based solely on gender. A perfectly unbiased model scores 0.

**Demographic Parity Gap** — measures the difference in overall predicted hire rates between male and female candidates across the full test set. Captures systematic skew toward one group even when individual predictions don't fully flip. A score of 0 means equal hire rates predicted for both genders.

# Final Results & Conclusion 
