import os
from src.load_data import load_subject
from src.windowing import create_windows
from src.build_dataset import build_dataset
import pandas as pd

DATA_PATH = "data/WESAD"

all_dfs = []

for subject in os.listdir(DATA_PATH):
    
    subject_path = os.path.join(DATA_PATH, subject)
    
    if not subject.startswith("S"):
        continue
    
    pkl_file = os.path.join(subject_path, f"{subject}.pkl")
    
    print(f"Processing {subject}...")
    #load, then windowing and then building the dataset
    chest, labels = load_subject(pkl_file)
    windows = create_windows(chest, labels)
    df = build_dataset(windows)
    
    df["Subject"] = subject  
    
    all_dfs.append(df)

# Combine all subjects
final_df = pd.concat(all_dfs, ignore_index=True)

# Save
final_df.to_csv("final_dataset.csv", index=False)

print("Final dataset shape:", final_df.shape)
print("DONE")