import pandas as pd

df = pd.read_csv("data/resume_features.csv", low_memory=False)
df["Gender"] = (df["Gender"].astype(str).str.strip().str.capitalize())
df_flipped = df.copy()

# Flip Gender
df_flipped["Gender"] = df_flipped["Gender"].map({ 
    "Male": "Female",
    "Female": "Male"
})

df_doubled = pd.concat([df, df_flipped], ignore_index=True) # Combine original and flipped
df_doubled = df_doubled.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle dataset

test_size = int(0.2 * len(df_doubled)) # 20% of dataset for testing

test_df = df_doubled.iloc[:test_size] # First 20% rows
train_df = df_doubled.iloc[test_size:] # Remaining 80% rows

train_df.to_csv("data/flipped_dataset.csv", index=False)
test_df.to_csv("data/test.csv", index=False)