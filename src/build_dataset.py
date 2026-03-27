import pandas as pd
from src.features import extract_features

def build_dataset(windows):
    
    rows = []
    
    for w in windows:
        feat = extract_features(w)
        rows.append(feat)
    
    df = pd.DataFrame(rows)
    
    return df