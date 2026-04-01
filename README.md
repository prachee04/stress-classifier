# Multimodal Passive Sensing for Stress Detection
### A Deployment-Aware Sensor Selection Framework Using Multi-Criteria Decision Analysis

> **WESAD Dataset · 15 Subjects · 20 Signal Combinations · 4 ML Models · Custom MCDA Framework**

(paste streamlit and dataset link here)
---

##  Project Summary

This project answers a deceptively simple question:

> **How many sensors do you actually need to detect stress — and which ones?**

Using the WESAD physiological dataset, we train and evaluate machine learning classifiers across **20 signal combinations** (single sensor → all 6 sensors), then rank them not just by accuracy but by a custom **Multi-Criteria Decision Analysis (MCDA)** framework that factors in hardware cost, wearability, battery life, and signal robustness.

The result is a **deployment-aware recommendation system** — the optimal sensor set changes depending on whether you're building a consumer wristband app, a mid-budget clinic tool, or a full research lab setup.

---

## 🔬 The 6 Physiological Signals

| Signal | Sensor | What It Measures | Key Features |
|--------|--------|-----------------|--------------|
| **HRV** | ECG (chest) | Beat-to-beat interval variability | SDNN, RMSSD, LF/HF ratio |
| **EDA** | Skin conductance (wrist/fingers) | Sweat gland activity | SCL_mean, SCR_count, SCR_amplitude |
| **ACC** | 3-axis accelerometer (wrist) | Physical movement & restlessness | Variance, entropy, peak count |
| **Respiration** | Chest belt | Breathing rate & depth | Resp_mean, Resp_std, breathing rate |
| **Temperature** | Thermistor (wrist) | Skin surface temperature | Temp_mean, rate of change, min/max |
| **EMG** | Electrodes (trapezius + zygomaticus) | Muscle electrical activity | RMS, MAV, zero-crossing rate |

---

## 💡 Research Hypotheses

| # | Hypothesis | Outcome |
|---|-----------|---------|
| H1 | No single signal achieves optimal accuracy — 2–3 signals needed | ✅ Confirmed |
| H2 | EDA will be the strongest unimodal signal | ✅ Confirmed |
| H3 | ACC will outperform expectations as a behavioral stress indicator | ✅ Confirmed |
| H4 | HRV will underperform despite being the most cited stress biomarker | ✅ Confirmed |
| H5 | Adding signals beyond 3 yields diminishing returns / hurts performance | ✅ Confirmed (Surprise) |
| H6 | The optimal sensor combo changes depending on deployment context | ✅ Confirmed |
| H7 | The MCDA winner shifts at least twice across the λ sweep (0.2→1.0) | ✅ Confirmed (3 shifts) |

---

## ⚙️ Pipeline & Execution

```
Raw WESAD Data (700Hz, per subject .pkl files)
        │
        ▼
 Load & Synchronize 6 signals + labels per timestamp
        │
        ▼
 Segment into 60-second windows
 (700 samples/sec × 60 sec = 42,000 samples/window)
        │
        ▼
 Feature Extraction per window
 (HRV features, EDA decomposition, ACC stats, etc.)
        │
        ▼
 Final Feature Matrix (~500 rows × 13 features + label)
        │
        ▼
 Train 4 ML Models × 20 Signal Combinations = 80 experiments
 (Subject-wise cross-validation — person-independent)
        │
        ▼
 Evaluate: F1_macro, Recall_stress, MCC
        │
        ▼
 MCDA Framework: Score each combo on Performance + Cost
        │
        ▼
 Sensitivity Analysis across λ ∈ {0.2, 0.4, 0.5, 0.6, 0.8, 1.0}
        │
        ▼
 Streamlit Dashboard — Interactive results explorer
```

### Feature Engineering Details

```python
# Each 60-second window produces these features:
features = {
    # HRV (from raw ECG)
    'SDNN': std_dev_of_rr_intervals,        # Low = stressed
    'RMSSD': beat_to_beat_fluctuation,      # Low = stressed

    # EDA (electrodermal)
    'SCL_mean': baseline_skin_conductance,
    'SCR_n': count_of_stress_spikes,
    'EDA_std': variation_in_sweat_signal,

    # Accelerometer
    'ACC_std': movement_variation,           # High = restless
    'ACC_entropy': movement_randomness,

    # Respiration
    'Resp_mean': avg_breathing_signal,
    'Resp_std': breathing_variation,

    # Temperature
    'Temp_mean': avg_skin_temp,             # Drops under stress

    # EMG
    'EMG_RMS': muscle_activation_level,

    'Label': {1: 'calm', 2: 'stress', 3: 'amusement'},
    'Subject': 'S2'  # Subject ID for cross-validation
}
```

### ML Models Used

| Model | Type | Rationale |
|-------|------|-----------|
| Logistic Regression | Linear | Interpretable baseline |
| Random Forest | Ensemble Tree | Handles non-linearity, robust to overfitting |
| XGBoost | Gradient Boosting | Best performance on tabular data |
| SVM (RBF kernel) | Margin-based | Excellent for imbalanced, high-dim data |

All models use `class_weight='balanced'` and **subject-wise cross-validation** (train on S2–S14, test on S15 etc.) to ensure person-independent generalization.

---

## 📊 Results

### Tier Summary

| Tier | Best Combination | Best Model | Performance Score |
|------|-----------------|------------|-------------------|
| Unimodal | **EDA** | SVM | 0.7346 |
| 2-signal | **EDA + ACC** | SVM | 0.7418 |
| 3-signal | **EDA + ACC + EMG** | XGBoost | 0.7440 |
| All 6 signals | — | RandomForest | 0.7150 ⚠️ |

> ⚠️ **All 6 signals (P=0.715) is outperformed by EDA+ACC+EMG (P=0.744).** More sensors ≠ better performance.



## 🎯 MCDA Framework

### Evaluation Metrics

```
Performance Score:
P = 0.40 × F1_macro + 0.35 × Recall_stress + 0.25 × MCC_norm

Cost Score (per signal, 1–5 scale):
C = 0.25×H + 0.25×W + 0.25×B + 0.25×(6−R)
    H = hardware cost | W = wearability | B = battery | R = robustness

Optimized Score:
O = λ × P + (1−λ) × (1 − Cost_norm)
```

### Sensitivity Analysis — λ Sweep

**The winner shifts 3 times: ACC → EDA+ACC → EDA+ACC+EMG**

---

## 🏆 Key Findings

1. **EDA is the best single sensor** — P=0.735, Recall_stress=0.839. Sweat gland activity is the most reliable physiological stress indicator.

2. **ACC is a surprisingly strong signal** — P=0.674 as a standalone. Behavioral patterns (restlessness, fidgeting) are powerful stress markers, consistent with Shukla et al. (2025).

3. **HRV underperforms despite being the most cited stress biomarker** — P=0.304. Motion artifacts at 700Hz sampling severely degrade HRV quality in real-world conditions.

4. **EDA+ACC is the sweet spot** — Best balance of performance (P=0.742) and cost (0.37). Both sensors are wristband-compatible and non-invasive.

5. **More sensors ≠ better performance** — All 6 signals (P=0.715) is WORSE than the 3-signal winner EDA+ACC+EMG (P=0.744). HRV and Respiration introduce noise that hurts the model.

6. **The optimal sensor set is deployment-dependent** — The MCDA winner changes 3 times across the λ sweep. This is the central contribution of this work.

---

## 📱 Deployment Recommendations

| Context | Best Combo | Optimized Score | Reasoning |
|---------|-----------|-----------------|-----------|
| **Consumer wearable app** (λ=0.2) | ACC alone | O=0.855 | Cheapest, most comfortable, already in every smartwatch |
| **Balanced real-world** (λ=0.5) | EDA+ACC | O=0.674 | Best performance-cost balance, both wristband-compatible |
| **Clinical/lab** (λ=1.0) | EDA+ACC+EMG | O=0.744 | Max stress recall, but requires gel electrodes on shoulder+face |


## 📁 Project Structure

```
├── data/
│   └── WESAD/                  # Raw dataset (S2–S17 subject folders)
├── notebooks/
│   ├── 01_feature_extraction.ipynb
│   ├── 02_unimodal_models.ipynb
│   ├── 03_multimodal_combinations.ipynb
│   └── 04_mcda_analysis.ipynb
├── src/
│   ├── feature_engineering.py
│   ├── models.py
│   └── mcda.py
├── streamlit_app/
│   └── app.py                  # Interactive dashboard
├── results/
│   └── experiment_results_summary.html
├── README.md
└── requirements.txt
```

---

**Requirements:** Python 3.9+, scikit-learn, xgboost, neurokit2, pandas, numpy, matplotlib, streamlit


---

*Built with the WESAD dataset. Inspired by Prof. Shukla's work on affective computing and passive digital biomarkers.*
