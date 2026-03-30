import pandas as pd


cost_df = pd.read_csv("cost_scores.csv")
perf_df = pd.read_csv("signal_performance.csv")

cost_df = cost_df[["Signal", "Cost_norm"]]
perf_df = perf_df[["Signal", "Performance"]]

df = pd.merge(perf_df, cost_df, on="Signal", how="inner")

print("\nMerged Data:")
print(df.head())

lambdas = [0.2, 0.4, 0.6, 0.8, 1.0]

for lam in lambdas:
    col_name = f"O_lambda_{lam}"
    
    df[col_name] = (
        lam * df["Performance"] +
        (1 - lam) * (1 - df["Cost_norm"])
    )

df.to_csv("optimized_results.csv", index=False)
print("\nSaved: optimized_results.csv")


best_rows = []

for lam in lambdas:
    col = f"O_lambda_{lam}"
    
    best_idx = df[col].idxmax()
    best_row = df.loc[best_idx]
    
    best_rows.append({
        "Lambda": lam,
        "Best_Signal": best_row["Signal"],
        "Optimized_Score": best_row[col],
        "Performance": best_row["Performance"],
        "Cost_norm": best_row["Cost_norm"]
    })

best_df = pd.DataFrame(best_rows)

best_df.to_csv("best_combinations_by_lambda.csv", index=False)

print("\nBest combinations per lambda:")
print(best_df)