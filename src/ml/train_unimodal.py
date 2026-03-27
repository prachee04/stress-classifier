import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import f1_score, recall_score, matthews_corrcoef

from xgboost import XGBClassifier


FEATURE_GROUPS = {
    "HRV": ["SDNN", "RMSSD"],
    "EDA": ["SCL_mean", "SCR_N", "EDA_std"],
    "ACC": ["ACC_std", "ACC_entropy"],
    "Resp": ["Resp_mean", "Resp_std"],
    "Temp": ["Temp_mean"],
    "EMG": ["EMG_RMS"]
}


def load_data():
    df = pd.read_csv("final_dataset.csv")
    
    # Keep only valid labels
    df = df[df["Label"].isin([1, 2, 3])]
    
    # Convert labels (1,2,3 → 0,1,2)
    df["Label"] = df["Label"].map({1: 0, 2: 1, 3: 2})
    
    df["Label"] = df["Label"].astype(int)
    
    return df


def subject_split(X, y, groups):
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def get_models():
    return {
        "Logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ),
        
        "SVM": SVC(class_weight="balanced"),
        
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42
        )
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    
    results = []
    models = get_models()
    
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    for name, model in models.items():
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Macro metrics
        f1 = f1_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        mcc = matthews_corrcoef(y_test, y_pred)
        
        per_class_recall = recall_score(y_test, y_pred, average=None)
        
        results.append({
            "Model": name,
            "F1_macro": f1,
            "Recall_macro": recall,
            "MCC": mcc,
            "Recall_class0": per_class_recall[0],  # baseline
            "Recall_class1": per_class_recall[1],  # stress
            "Recall_class2": per_class_recall[2]   # amusement
        })
    
    return pd.DataFrame(results)


def run_unimodal():
    
    df = load_data()
    
    print("Dataset shape:", df.shape)
    print("Label distribution:\n", df["Label"].value_counts())
    
    all_results = []
    
    for signal, features in FEATURE_GROUPS.items():
        
        print(f"\n==============================")
        print(f"Running for {signal}")
        print(f"Features: {features}")
        
        X = df[features]
        y = df["Label"]
        groups = df["Subject"]
        
        X_train, X_test, y_train, y_test = subject_split(X, y, groups)
        
        result_df = train_and_evaluate(X_train, X_test, y_train, y_test)
        result_df["Signal"] = signal
        
        print(result_df)
        
        all_results.append(result_df)
    
    final_results = pd.concat(all_results, ignore_index=True)
    
    final_results.to_csv("unimodal_results.csv", index=False)
    
    print("\nResults saved to unimodal_results.csv")


if __name__ == "__main__":
    run_unimodal()