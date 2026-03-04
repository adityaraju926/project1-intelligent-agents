import pandas as pd

df = pd.read_csv("data/resume_features.csv", low_memory=False)
df["Gender"] = (df["Gender"].astype(str).str.strip().str.capitalize())
df_flipped = df.copy()

# Flip Gender
df_flipped["Gender"] = df_flipped["Gender"].map({ 
    "Male": "Female",
    "Female": "Male"
})

# Combine original and flipped
df_doubled = pd.concat([df, df_flipped], ignore_index=True)

# Shuffle dataset
df_doubled = df_doubled.sample(frac=1, random_state=42).reset_index(drop=True)

df_doubled.to_csv("data/flipped_dataset.csv", index=False)