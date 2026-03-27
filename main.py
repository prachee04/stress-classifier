import os
import pandas as pd

from src.load_data import load_subject
from src.windowing import create_windows
from src.build_dataset import build_dataset

DATA_PATH = "data/WESAD"

all_dfs = []

for subject in os.listdir(DATA_PATH):
    
    if not subject.startswith("S"):
        continue
    
    subject_path = os.path.join(DATA_PATH, subject)
    pkl_file = os.path.join(subject_path, f"{subject}.pkl")
    
    print(f"Processing {subject}...")
    
    try:
        chest, labels = load_subject(pkl_file)
        
        windows = create_windows(chest, labels)
        print(f"{subject} → windows: {len(windows)}")
        
        df = build_dataset(windows)
        
        # Add subject column
        df["Subject"] = subject
        
        all_dfs.append(df)
    
    except Exception as e:
        print(f"Error processing {subject}: {e}")

# Combine all subjects
final_df = pd.concat(all_dfs, ignore_index=True)

# Save dataset
final_df.to_csv("final_dataset.csv", index=False)

print("\nFinal dataset shape:", final_df.shape)
print("DONE")