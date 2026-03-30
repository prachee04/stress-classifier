import pandas as pd
import numpy as np


# H = Hardware cost 
# W = Wearability       
# B = Battery drain      
# R = Robustness         


SIGNAL_SCORES = {
    #         H   W   B   R
    "HRV":  { "H": 3, "W": 3, "B": 3, "R": 3 },
    "EDA":  { "H": 4, "W": 3, "B": 3, "R": 3 },
    "ACC":  { "H": 1, "W": 1, "B": 1, "R": 5 },
    "Resp": { "H": 3, "W": 4, "B": 3, "R": 2 },
    "Temp": { "H": 1, "W": 1, "B": 1, "R": 5 },
    "EMG":  { "H": 5, "W": 5, "B": 5, "R": 2 },
}


ALPHA = BETA = GAMMA = DELTA = 0.25

COMBINATIONS = [
    # Unimodal
    ["HRV"],
    ["EDA"],
    ["ACC"],
    ["Resp"],
    ["Temp"],
    ["EMG"],
    # 2-signal
    ["HRV", "EDA"],
    ["HRV", "ACC"],
    ["HRV", "Resp"],
    ["HRV", "EMG"],
    ["EDA", "ACC"],
    ["EDA", "Temp"],
    ["EDA", "Resp"],
    ["ACC", "EMG"],
    # 3-signal
    ["HRV", "EDA", "ACC"],
    ["HRV", "EDA", "Resp"],
    ["EDA", "ACC", "EMG"],
    ["HRV", "ACC", "EMG"],
    ["EDA", "Temp", "Resp"],
    # All 6
    ["HRV", "EDA", "ACC", "Resp", "Temp", "EMG"],
]


def signal_cost(signal):
    #Cᵢ = αH + βW + γB + δ(6−R)
    s = SIGNAL_SCORES[signal]
    return (
        ALPHA * s["H"] +
        BETA  * s["W"] +
        GAMMA * s["B"] +
        DELTA * (6 - s["R"])
    )


def combination_cost(signals):

    return np.mean([signal_cost(s) for s in signals])


def compute_all_costs():
    rows = []
    for combo in COMBINATIONS:
        name     = "+".join(combo)
        n        = len(combo)
        raw_cost = combination_cost(combo)

        row = {
            "Signal":   name,
            "n_signals": n,
            "Raw_Cost": round(raw_cost, 4),
        }

        # Individual signal costs
        for s in combo:
            row[f"C_{s}"] = round(signal_cost(s), 4)

        rows.append(row)

    df = pd.DataFrame(rows)

    #normalisation
    min_c = df["Raw_Cost"].min()
    max_c = df["Raw_Cost"].max()
    df["Cost_norm"] = 0.1 + 0.9 * (
    (df["Raw_Cost"] - min_c) / (max_c - min_c)
)

    df["Cost_norm"] = df["Cost_norm"].round(5)
    return df


if __name__ == "__main__":
    df = compute_all_costs()

    print("=" * 55)
    print("  SIGNAL COST SCORES (individual)")
    print("=" * 55)
    for sig, scores in SIGNAL_SCORES.items():
        c = signal_cost(sig)
        print(f"  {sig:<6}  H={scores['H']} W={scores['W']} "
              f"B={scores['B']} R={scores['R']}  →  Cᵢ = {c:.4f}")

    print("\n" + "=" * 55)
    print("  ALL COMBINATIONS, raw and normalised cost")
    print("=" * 55)
    print(df[["Signal", "n_signals", "Raw_Cost", "Cost_norm"]].to_string(index=False))

    df.to_csv("cost_scores.csv", index=False, float_format="%.5f")
    print("\nSaved at cost_scores.csv")