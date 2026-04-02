# Multimodal Passive Sensing for Stress Detection

### A deployment-aware sensor selection framework using MCDA

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-WESAD-green.svg)](https://ubicomp.net/sw/data/wesad.html)

> **Live dashboard →** `https://stress-classifier-awaqcc2kv9e2c47jm8vlcd.streamlit.app/`  
> **Project summary →** `[link to summary doc]`  
> **Contact →** `mprachee2204@gmail.com`

---

## What makes this different

Every WESAD paper asks: *does combining sensors improve accuracy?* The answer is always yes.

This study asks the harder question: **which combination is actually worth deploying — given that sensors have real hardware costs, wearability constraints, battery limits, and signal quality issues?**

The answer depends entirely on who is deploying it. This framework quantifies that tradeoff.

---

## The core finding

The optimal sensor combination **changes** as deployment context changes:

| Context | Winner | Optimised Score |
|---------|--------|----------------|
| Consumer app (λ=0.2) | **ACC alone** | O = 0.855 |
| Balanced deployment (λ=0.5) | **EDA + ACC** | O = 0.674 |
| Lab / clinical (λ=1.0) | **EDA + ACC + EMG** | O = 0.744 |

And the most surprising finding: **all 6 signals together (P=0.715) performs worse than the best 3-signal combination (P=0.744)**. More sensors ≠ better performance.

---

## Dataset

**WESAD** (Wearable Stress and Affect Detection)  
Schmidt et al., ICMI 2018

- 15 subjects, chest-worn device
- 700 Hz sampling rate
- 6 physiological signals: ECG, EDA, EMG, Respiration, Temperature, ACC
- Labels: 1=calm, 2=stress, 3=amusement → remapped to 0/1/2
- ~500 windows × 60 seconds after windowing

---

## The 6 signals

| Signal | Measures | Sensor platform | Cost tier |
|--------|----------|----------------|-----------|
| **ECG (HRV)** | Heart rate variability — gaps between beats | Shimmer3R ECG | Medium |
| **EDA** | Sweat gland activity — sympathetic arousal | Empatica E4 / EmbracePlus | High |
| **ACC** | Physical movement and restlessness | MEMS chip (any wristband) | Very low |
| **Respiration** | Breathing rate and depth | Shimmer3 chest belt | Medium |
| **Temperature** | Skin surface temperature | NTC thermistor | Very low |
| **EMG** | Muscle tension — trapezius and zygomaticus | Shimmer3R EMG / Bitalino | Very high |

---

## Research hypotheses

Six hypotheses were tested. Results after full experiment:

| # | Hypothesis | Result |
|---|-----------|--------|
| H1 | Multimodal outperforms unimodal | ✅ **Confirmed** |
| H2 | All 6 signals produce the highest accuracy | ❌ **Rejected** — 3-signal wins |
| H3 | EDA is the strongest single signal | ✅ **Confirmed** — P=0.735 |
| H4 | Optimal combination shifts with deployment context | ✅ **Confirmed** — winner changes 3× |
| H5 | HRV is a strong stress indicator | ❌ **Rejected** — weakest signal (P=0.304) |
| H6 | A 2-sensor wristband can compete with full lab setup | ✅ **Confirmed** — EDA+ACC is 0.002 behind best 3-signal |

---

## Project structure

```
stress-classifier/
├── data/
│   └── wesad/
│       └── S2, S3, S4 ... S17/      ← WESAD raw .pkl files
├── src/
│   ├── load_data.py                  ← Load .pkl, extract chest signals + labels
│   ├── windowing.py                  ← Segment into 60s windows at 700Hz
│   ├── build_dataset.py             ← Orchestrate windowing → feature extraction
│   ├── features.py                   ← Feature extraction per signal per window
│   └── ml/
│       ├── train_unimodal.py         ← Train 4 models × 6 signals
│       └── train_multimodal.py       ← Train 4 models × 13 combinations
├── main.py                           ← Run full pipeline → final_dataset.csv
├── compute_cost.py                   ← MCDA cost calculation for all 20 combos
├── app.py                  ← Interactive results dashboard
├── final_dataset.csv                 ← Feature matrix (500 rows × 13 cols)
├── unimodal_results.csv             ← Results for 6 unimodal signals
├── multimodal_results.csv           ← Results for 13 multimodal combos
└── cost_scores.csv                  ← MCDA cost scores for all 20 combos
```

---

## Pipeline

```
Raw WESAD .pkl
      │
      ▼
load_data.py      → Extract 6 signals + labels per subject
      │
      ▼
windowing.py      → 60s windows (42,000 samples each at 700Hz)
      │
      ▼
features.py       → 11 features: SDNN, RMSSD, SCL_mean, SCR_N, EDA_std,
      │               ACC_std, ACC_entropy, Resp_mean, Resp_std, Temp_mean, EMG_RMS
      ▼
final_dataset.csv → ~500 rows × 13 cols (features + Label + Subject)
      │
      ├──► train_unimodal.py    → unimodal_results.csv
      └──► train_multimodal.py  → multimodal_results.csv
                │
                ▼
          compute_cost.py       → cost_scores.csv
                │
                ▼
          Excel / Python        → Performance score P, Optimised score O per λ
                │
                ▼
          streamlit_app.py      → Interactive dashboard
```

---

## Features extracted

| Feature | Signal | What it captures |
|---------|--------|-----------------|
| SDNN | HRV | Standard deviation of heartbeat intervals — lower = more stressed |
| RMSSD | HRV | Short-term beat-to-beat variation — lower = more stressed |
| SCL_mean | EDA | Baseline skin conductance level |
| SCR_N | EDA | Number of stress spikes in the window |
| EDA_std | EDA | Variability in skin conductance |
| ACC_std | ACC | Movement variability — fidgeting |
| ACC_entropy | ACC | Randomness of movement pattern |
| Resp_mean | Respiration | Average breathing signal amplitude |
| Resp_std | Respiration | Breathing variability |
| Temp_mean | Temperature | Average skin temperature |
| EMG_RMS | EMG | Muscle activation level |

---

## ML models

| Model | Type | Why included |
|-------|------|-------------|
| Logistic Regression | Linear | Interpretable baseline |
| Random Forest | Tree ensemble | Non-linear, handles feature interactions |
| SVM | Margin-based | Strong for small datasets with high-dim features |
| XGBoost | Gradient boosting | Best performance on tabular data generally |

All models use `class_weight="balanced"` to handle class imbalance.  
Train/test split: `GroupShuffleSplit(test_size=0.2)` — no subject in both train and test.

---

## Evaluation metrics

**Headline metrics (used in model selection):**

```
Performance P = 0.40 · F1_macro  +  0.35 · Recall_stress  +  0.25 · MCC_norm

where MCC_norm = (MCC + 1) / 2    →  range [0, 1]
```

**Supplementary (reported in full results table):** Specificity, AUC-ROC (OvR), Cohen's Kappa

**Why Recall_stress gets the highest weight:** In clinical mental health applications, missing a real stress episode (false negative) is worse than a false alarm. This asymmetric weighting reflects the Fβ framework with β>1.

---

## MCDA cost scoring

Each signal scored on 4 dimensions sourced from published device datasheets:

```
Cᵢ = 0.25·H + 0.25·W + 0.25·B + 0.25·(6−R)

where:
  H = Hardware cost    (1=cheap, 5=expensive)
  W = Wearability      (1=comfortable, 5=intrusive)
  B = Battery drain    (1=low, 5=high)
  R = Robustness       (1=fragile, 5=robust) — inverted via (6−R)

For combinations: C_combo = (1/n) Σ Cᵢ   [arithmetic mean]
Normalised:       C_norm  = (C − min) / (max − min)  →  [0.1, 1.0]
```

**Sources:** iMotions (2025), Shimmer Research datasheets, Empatica E4 specs,  
Bitalino EMG specs, MDPI motion artifact review (2020), npj Cardiovascular Health (2025)

---

## Optimised score

```
O = λ · P  +  (1 − λ) · (1 − C_norm)

λ = 0.2  →  Consumer app (prioritise cheapness and comfort)
λ = 0.5  →  Balanced deployment
λ = 1.0  →  Lab / clinical (ignore cost, maximise performance)
```

Sensitivity analysis run at λ ∈ {0.2, 0.4, 0.6, 0.8, 1.0}.

---

## Future work

- **Phase 2 — Anxiety/depression classification:** Apply pipeline to K-EmoCon, DREAMER, or MAHNOB-HCI datasets with valence-arousal labels. Map EDA+ACC signal pair to the Russell circumplex model.

- **Phase 3 — Personality-modulated classification:** Integrate Big Five (OCEAN) personality scores as features. Neuroticism modulates EDA reactivity; Conscientiousness modulates HRV regulation. ASCERTAIN dataset supports this directly.

- **Phase 4 — Clinical disorder classification:** Extend sensor selection framework to anxiety disorders, OCD, BPD, PTSD. The MCDA approach generalises to any condition where sensor cost tradeoffs matter.

---

## Citations

1. Chicco & Jurman (2020). MCC advantages over F1 and accuracy. *BMC Genomics*, 21(1), 6.
2. Chicco, Tötsch & Jurman (2021). MCC reliability. *BioData Mining*, 14, 13.
3. Ivlev et al. (2016). MCDA for MRI selection. *Int. J. Medical Engineering and Informatics*, 8(2).
4. Ivlev et al. (2015). MCDA for medical devices under uncertainty. *EJOR*, 247(1).
5. Marsh et al. (2016). MCDA emerging good practices. *Value in Health*, 19(2). PMC4828475.
6. Shbool et al. (2021). MCDA framework for medical device selection. *Cogent Engineering*, 8(1).
7. Luo et al. (2021). Performance metrics for ML models. *Radiology: AI*. PMC8204137.
8. Ling, Huang & Zhang (2008). ROC curves and Cohen's kappa. *Engineering Applications of AI*.
9. Yadav, Upadhyay & Shukla (2025). Multimodal biomarkers for ADHD. *TechRxiv*. DOI: 10.36227/techrxiv.176403977.
10. Schmidt et al. (2018). Introducing WESAD. *ICMI 2018*.
