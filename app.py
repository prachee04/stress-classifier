import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

st.set_page_config(
    page_title="Stress & Anxiety Sensor Study",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

UNIMODAL = pd.DataFrame([
    {"Signal":"HRV",  "Model":"Logistic",     "F1_macro":0.2401,"Recall_macro":0.3106,"MCC":-0.0556,"Recall_class1":0.2581,"MCC_norm":0.4722,"Performance":0.3044,"Type":"Unimodal"},
    {"Signal":"EDA",  "Model":"SVM",          "F1_macro":0.6020,"Recall_macro":0.6117,"MCC":0.6020, "Recall_class1":0.8387,"MCC_norm":0.8010,"Performance":0.7346,"Type":"Unimodal"},
    {"Signal":"ACC",  "Model":"XGBoost",      "F1_macro":0.5601,"Recall_macro":0.5598,"MCC":0.4334, "Recall_class1":0.7742,"MCC_norm":0.7167,"Performance":0.6742,"Type":"Unimodal"},
    {"Signal":"Resp", "Model":"Logistic",     "F1_macro":0.4667,"Recall_macro":0.4486,"MCC":0.3071, "Recall_class1":0.4194,"MCC_norm":0.6535,"Performance":0.4968,"Type":"Unimodal"},
    {"Signal":"Temp", "Model":"XGBoost",      "F1_macro":0.3723,"Recall_macro":0.3744,"MCC":0.0999, "Recall_class1":0.3548,"MCC_norm":0.5500,"Performance":0.4106,"Type":"Unimodal"},
    {"Signal":"EMG",  "Model":"XGBoost",      "F1_macro":0.4126,"Recall_macro":0.4259,"MCC":0.1078, "Recall_class1":0.1935,"MCC_norm":0.5539,"Performance":0.3713,"Type":"Unimodal"},
])

MULTIMODAL = pd.DataFrame([
    {"Signal":"HRV+EDA",         "Model":"SVM",        "F1_macro":0.5568,"Recall_macro":0.5491,"MCC":0.4207,"Recall_class1":0.7419,"MCC_norm":0.7103,"Performance":0.6600,"Type":"2-signal","Cost_norm":0.61},
    {"Signal":"HRV+ACC",         "Model":"RandomForest","F1_macro":0.4111,"Recall_macro":0.4155,"MCC":0.1545,"Recall_class1":0.5484,"MCC_norm":0.5773,"Performance":0.5007,"Type":"2-signal","Cost_norm":0.34},
    {"Signal":"HRV+Resp",        "Model":"Logistic",   "F1_macro":0.3575,"Recall_macro":0.3693,"MCC":0.2191,"Recall_class1":0.1290,"MCC_norm":0.6096,"Performance":0.3405,"Type":"2-signal","Cost_norm":0.64},
    {"Signal":"HRV+EMG",         "Model":"XGBoost",    "F1_macro":0.2983,"Recall_macro":0.3361,"MCC":-0.0727,"Recall_class1":0.2258,"MCC_norm":0.4637,"Performance":0.3143,"Type":"2-signal","Cost_norm":0.79},
    {"Signal":"EDA+ACC",         "Model":"SVM",        "F1_macro":0.6361,"Recall_macro":0.6269,"MCC":0.5505,"Recall_class1":0.8387,"MCC_norm":0.7753,"Performance":0.7418,"Type":"2-signal","Cost_norm":0.37},
    {"Signal":"EDA+Temp",        "Model":"SVM",        "F1_macro":0.5591,"Recall_macro":0.5837,"MCC":0.5647,"Recall_class1":0.8387,"MCC_norm":0.7823,"Performance":0.7128,"Type":"2-signal","Cost_norm":0.37},
    {"Signal":"EDA+Resp",        "Model":"SVM",        "F1_macro":0.5175,"Recall_macro":0.5142,"MCC":0.4573,"Recall_class1":0.6129,"MCC_norm":0.7286,"Performance":0.6037,"Type":"2-signal","Cost_norm":0.67},
    {"Signal":"ACC+EMG",         "Model":"XGBoost",    "F1_macro":0.5599,"Recall_macro":0.5898,"MCC":0.3667,"Recall_class1":0.7097,"MCC_norm":0.6833,"Performance":0.6432,"Type":"2-signal","Cost_norm":0.55},
    {"Signal":"HRV+EDA+ACC",     "Model":"SVM",        "F1_macro":0.6342,"Recall_macro":0.6267,"MCC":0.5209,"Recall_class1":0.8065,"MCC_norm":0.7605,"Performance":0.7261,"Type":"3-signal","Cost_norm":0.44},
    {"Signal":"HRV+EDA+Resp",    "Model":"RandomForest","F1_macro":0.4876,"Recall_macro":0.5133,"MCC":0.4260,"Recall_class1":0.6452,"MCC_norm":0.7130,"Performance":0.5991,"Type":"3-signal","Cost_norm":0.64},
    {"Signal":"EDA+ACC+EMG",     "Model":"XGBoost",    "F1_macro":0.6498,"Recall_macro":0.6527,"MCC":0.5239,"Recall_class1":0.8387,"MCC_norm":0.7619,"Performance":0.7440,"Type":"3-signal","Cost_norm":0.58},
    {"Signal":"HRV+ACC+EMG",     "Model":"RandomForest","F1_macro":0.4347,"Recall_macro":0.4670,"MCC":0.1507,"Recall_class1":0.5484,"MCC_norm":0.5754,"Performance":0.5097,"Type":"3-signal","Cost_norm":0.56},
    {"Signal":"EDA+Temp+Resp",   "Model":"RandomForest","F1_macro":0.5338,"Recall_macro":0.5533,"MCC":0.5547,"Recall_class1":0.6774,"MCC_norm":0.7773,"Performance":0.6449,"Type":"3-signal","Cost_norm":0.48},
    {"Signal":"All 6 signals",   "Model":"RandomForest","F1_macro":0.6039,"Recall_macro":0.5998,"MCC":0.5295,"Recall_class1":0.8065,"MCC_norm":0.7648,"Performance":0.7150,"Type":"6-signal","Cost_norm":0.52},
])

UNIMODAL["Cost_norm"] = [0.58, 0.64, 0.10, 0.70, 0.10, 1.00]
ALL_DATA = pd.concat([UNIMODAL, MULTIMODAL], ignore_index=True)

LAMBDAS = [0.2, 0.4, 0.6, 0.8, 1.0]
for lam in LAMBDAS:
    ALL_DATA[f"O_{lam}"] = (lam * ALL_DATA["Performance"] + (1-lam) * (1 - ALL_DATA["Cost_norm"])).round(4)

TYPE_COLORS = {
    "Unimodal": "#888780",
    "2-signal": "#378ADD",
    "3-signal": "#1D9E75",
    "6-signal": "#7F77DD",
}

SIGNAL_INFO = {
    "HRV":  {"platform":"Shimmer3R ECG","H":3,"W":3,"B":3,"R":3,"placement":"Chest electrodes"},
    "EDA":  {"platform":"Empatica E4","H":4,"W":3,"B":3,"R":3,"placement":"Wrist / fingers"},
    "ACC":  {"platform":"MEMS chip (any wristband)","H":1,"W":1,"B":1,"R":5,"placement":"Wrist"},
    "Resp": {"platform":"Shimmer3 chest belt","H":3,"W":4,"B":3,"R":2,"placement":"Chest belt"},
    "Temp": {"platform":"NTC thermistor","H":1,"W":1,"B":1,"R":5,"placement":"Wrist skin"},
    "EMG":  {"platform":"Shimmer3R EMG / Bitalino","H":5,"W":5,"B":5,"R":2,"placement":"Trapezius + zygomaticus"},
}

# SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "Overview",
    "λ Explorer",
    "Ablation Study",
    # "Sensor Recommender",
    # "Chart Code",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** WESAD (15 subjects)")
st.sidebar.markdown("**Signals:** 6 physiological")
st.sidebar.markdown("**Windows:** ~500 × 60s")
st.sidebar.markdown("**Classes:** Calm / Stress / Amusement")

# PAGE 1 — OVERVIEW

if page == "Overview":
    st.title("Multimodal Passive Sensing for Stress Detection")
    st.markdown("**A deployment-aware sensor selection framework using MCDA**")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best unimodal", "EDA", "P = 0.735")
    c2.metric("Best 2-signal", "EDA + ACC", "P = 0.742")
    c3.metric("Best 3-signal", "EDA+ACC+EMG", "P = 0.744")
    c4.metric("All 6 signals", "P = 0.715", "↓ worse than 3-signal")

    st.markdown("---")
    st.subheader("The core question")
    st.info("Every existing WESAD paper asks: *does multimodal beat unimodal?*\n\nThis study asks something harder: **what is the minimum viable sensor set — balancing classification performance against real-world deployment cost?**\n\nThe answer depends on who is deploying it.")

    st.markdown("---")
    st.subheader("The 6 signals")
    cols = st.columns(3)
    for i, (sig, info) in enumerate(SIGNAL_INFO.items()):
        with cols[i % 3]:
            cost_raw = 0.25*info["H"] + 0.25*info["W"] + 0.25*info["B"] + 0.25*(6-info["R"])
            st.markdown(f"**{sig}**")
            st.markdown(f"Platform: {info['platform']}")
            st.markdown(f"Placement: {info['placement']}")
            st.markdown(f"Raw cost: `{cost_raw:.2f}` / 5")
            st.markdown("---")

    st.subheader("Key findings at a glance")
    findings = [
        ("Consumer app (λ=0.2)", "ACC alone wins", "O=0.855 — wristwatch chip, no setup, days of battery."),
        ("Balanced deployment (λ=0.5)", "EDA + ACC wins", "O=0.674 — two wristband sensors, best performance-cost tradeoff."),
        ("Lab / clinical (λ=1.0)", "EDA+ACC+EMG wins", "O=0.744 — adding EMG boosts stress recall but triples setup complexity."),
        ("Surprise finding", "More ≠ Better", "All 6 signals (O=0.715) is WORSE than EDA+ACC+EMG (0.744). Noise from weak signals hurts."),
    ]
    for title, winner, detail in findings:
        with st.expander(f"{title}  →  **{winner}**"):
            st.write(detail)


elif page == "Ablation Study":
    st.title("Ablation Study")
    st.markdown("Are all 6 signals required? Is 1 enough? What is the optimal combination?")
    st.markdown("---")

    sort_by = st.selectbox("Sort by", ["Performance", "F1_macro", "Recall_class1", "MCC"])
    show_types = st.multiselect("Show signal types", ["Unimodal", "2-signal", "3-signal", "6-signal"],
                                 default=["Unimodal", "2-signal", "3-signal", "6-signal"])

    df_plot = ALL_DATA[ALL_DATA["Type"].isin(show_types)].sort_values(sort_by, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot)*0.4)))
    bar_colors = [TYPE_COLORS[t] for t in df_plot["Type"]]
    bars = ax.barh(df_plot["Signal"], df_plot[sort_by], color=bar_colors, height=0.6)

    for bar, val in zip(bars, df_plot[sort_by]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlabel(sort_by.replace("_", " "), fontsize=11)
    ax.set_xlim(0, df_plot[sort_by].max() * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    legend_patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items() if t in show_types]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Ablation table — all 20 combinations")
    display_cols = ["Signal", "Type", "Model", "F1_macro", "Recall_class1", "MCC", "Performance", "Cost_norm"]
    st.dataframe(
        ALL_DATA[display_cols].sort_values("Performance", ascending=False).reset_index(drop=True).style.format({
            c: "{:.4f}" for c in ["F1_macro","Recall_class1","MCC","Performance","Cost_norm"]
        }).background_gradient(subset=["Performance"], cmap="Greens"),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Key insights")
    st.success("**EDA and ACC are the two most important signals.** EDA alone achieves P=0.735 — best unimodal. Adding ACC to EDA (EDA+ACC) raises it to 0.742 for marginal cost increase.")
    st.warning("**Diminishing returns above 2 signals.** The jump from 1→2 signals is +0.008 (EDA→EDA+ACC). The jump from 2→3 is only +0.002 (EDA+ACC→EDA+ACC+EMG). Not worth the cost in most deployments.")
    st.error("**All 6 signals is not the best model.** Performance = 0.715, which is worse than EDA+ACC+EMG (0.744). HRV and Respiration introduce noise that hurts the classifier.")



elif page == "λ Explorer":
    st.title("λ Explorer — Deployment Context Optimisation")
    st.markdown("The optimal sensor combination is **not fixed** — it shifts depending on who is deploying it.")
    st.markdown("---")

    lam = st.slider("λ — deployment context", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                     help="λ=0 = prioritise cost (consumer app). λ=1 = prioritise performance (research lab).")

    col_desc = {0.0:"Consumer app — cheapest sensors only", 0.1:"Consumer app — cheapest sensors only",
                0.2:"Consumer app — cheapest sensors only", 0.3:"Balanced, lean practical",
                0.4:"Balanced, lean practical", 0.5:"Balanced deployment",
                0.6:"Balanced, lean performance", 0.7:"Balanced, lean performance",
                0.8:"Research lab — performance priority", 0.9:"Research lab — performance priority",
                1.0:"Pure performance — ignore cost"}
    st.caption(f"Context: {col_desc.get(round(lam,1), 'Balanced')}")

    ALL_DATA["O_current"] = (lam * ALL_DATA["Performance"] + (1-lam)*(1-ALL_DATA["Cost_norm"])).round(4)
    df_ranked = ALL_DATA.sort_values("O_current", ascending=False).reset_index(drop=True)
    winner = df_ranked.iloc[0]

    st.markdown("---")
    st.subheader("Current winner")
    w1, w2, w3 = st.columns(3)
    w1.metric("Signal combination", winner["Signal"])
    w2.metric("Optimised score O", f"{winner['O_current']:.4f}")
    w3.metric("Best ML model", winner["Model"])

    st.markdown("---")
    top_n = st.slider("Show top N combinations", min_value=5, max_value=20, value=10)
    df_top = df_ranked.head(top_n)

    fig2, ax2 = plt.subplots(figsize=(10, max(5, top_n*0.45)))
    bar_colors2 = [TYPE_COLORS[t] for t in df_top["Type"]]
    bars2 = ax2.barh(df_top["Signal"][::-1], df_top["O_current"][::-1], color=bar_colors2[::-1], height=0.6)
    bars2[0].set_edgecolor("#1a1a1a")
    bars2[0].set_linewidth(2)

    for bar, val in zip(bars2, df_top["O_current"][::-1]):
        ax2.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)

    ax2.set_xlabel(f"Optimised score O  (λ={lam:.1f})", fontsize=11)
    ax2.set_xlim(0, df_top["O_current"].max() * 1.12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.3)
    patches2 = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
    ax2.legend(handles=patches2, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    st.subheader("How winners change across all λ values")

    highlight = st.multiselect(
        "Highlight specific combinations",
        options=ALL_DATA["Signal"].tolist(),
        default=["ACC", "EDA+ACC", "EDA+ACC+EMG", "All 6 signals", "EDA"]
    )

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    lam_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    palette = plt.cm.tab10.colors

    for i, sig in enumerate(highlight):
        row = ALL_DATA[ALL_DATA["Signal"] == sig]
        if row.empty:
            continue
        row = row.iloc[0]
        y = [lam_v * row["Performance"] + (1-lam_v)*(1-row["Cost_norm"]) for lam_v in lam_vals]
        ax3.plot(lam_vals, y, marker="o", label=sig, color=palette[i % len(palette)], linewidth=2)
        ax3.annotate(sig, xy=(lam_vals[-1], y[-1]), xytext=(5, 0),
                     textcoords="offset points", fontsize=8, color=palette[i % len(palette)])

    ax3.axvline(x=lam, color="gray", linestyle="--", alpha=0.5, label=f"Current λ={lam:.1f}")
    ax3.set_xlabel("λ", fontsize=11)
    ax3.set_ylabel("Optimised score O", fontsize=11)
    ax3.set_xlim(0.15, 1.08)
    ax3.set_ylim(0.0, 1.0)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(alpha=0.2)
    ax3.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()



elif page == "Sensor Recommender":
    st.title("Sensor Recommender")
    st.markdown("Given your deployment context, which sensor combination should you use?")
    st.markdown("---")

    scenario = st.radio("Select your deployment scenario", [
        "Consumer wearable app (budget wristband)",
        "Balanced — clinical research tool",
        "Lab / hospital (no cost constraints)",
        "Custom — set my own λ",
    ])

    if scenario == "Consumer wearable app (budget wristband)":
        lam_rec = 0.2
    elif scenario == "Balanced — clinical research tool":
        lam_rec = 0.5
    elif scenario == "Lab / hospital (no cost constraints)":
        lam_rec = 1.0
    else:
        lam_rec = st.slider("Custom λ", 0.0, 1.0, 0.5, 0.05)

    ALL_DATA["O_rec"] = (lam_rec * ALL_DATA["Performance"] + (1-lam_rec)*(1-ALL_DATA["Cost_norm"])).round(4)
    best = ALL_DATA.sort_values("O_rec", ascending=False).iloc[0]
    top3 = ALL_DATA.sort_values("O_rec", ascending=False).head(3)

    st.markdown("---")
    st.subheader(f"Recommended combination at λ={lam_rec:.1f}")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Combination", best["Signal"])
    r2.metric("O score", f"{best['O_rec']:.4f}")
    r3.metric("Performance", f"{best['Performance']:.4f}")
    r4.metric("Cost (normalised)", f"{best['Cost_norm']:.2f}")

    st.markdown("---")
    st.subheader("Top 3 options")
    for _, row in top3.iterrows():
        sigs = row["Signal"].split("+")
        with st.expander(f"**{row['Signal']}**  |  O={row['O_rec']:.4f}  |  {row['Type']}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Performance", f"{row['Performance']:.4f}")
            c2.metric("Cost (norm)", f"{row['Cost_norm']:.2f}")
            c3.metric("Best model", row["Model"])
            st.markdown("**Signals in this combination:**")
            for sig in sigs:
                sig = sig.strip()
                if sig in SIGNAL_INFO:
                    info = SIGNAL_INFO[sig]
                    st.markdown(f"- **{sig}**: {info['platform']} — {info['placement']}")

    st.markdown("---")
    st.subheader("Cost breakdown of recommended combination")
    sigs_best = [s.strip() for s in best["Signal"].split("+") if s.strip() in SIGNAL_INFO]
    if sigs_best:
        cost_df = pd.DataFrame([
            {"Signal": s, "H": SIGNAL_INFO[s]["H"], "W": SIGNAL_INFO[s]["W"],
             "B": SIGNAL_INFO[s]["B"], "R (robustness)": SIGNAL_INFO[s]["R"],
             "Raw Cost": round(0.25*SIGNAL_INFO[s]["H"] + 0.25*SIGNAL_INFO[s]["W"] +
                               0.25*SIGNAL_INFO[s]["B"] + 0.25*(6-SIGNAL_INFO[s]["R"]), 2)}
            for s in sigs_best
        ])
        st.dataframe(cost_df, use_container_width=True)


elif page == "Chart Code":
    st.title("Research Paper Chart Code")
    st.markdown("Matplotlib/Seaborn code ready to copy into your paper workflow.")
    st.markdown("---")

    chart_choice = st.selectbox("Select chart", [
        "Chart 1 — Performance bar chart (all 20 combinations)",
        "Chart 2 — Lambda sensitivity line chart",
        "Chart 3 — Performance vs Cost scatter",
        "Chart 4 — Unimodal comparison bar chart",
        "Chart 5 — Winner heatmap across λ values",
    ])

    if chart_choice.startswith("Chart 1"):
        st.subheader("Chart 1 — Performance bar chart")
        st.code('''
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Load your results
df = pd.read_csv("multimodal_results.csv")   # adjust path
# Add unimodal too, then concat

TYPE_COLORS = {
    "Unimodal": "#888780",
    "2-signal": "#378ADD",
    "3-signal": "#1D9E75",
    "6-signal": "#7F77DD",
}

df_sorted = df.sort_values("Performance", ascending=True)
bar_colors = [TYPE_COLORS[t] for t in df_sorted["Type"]]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(df_sorted["Signal"], df_sorted["Performance"],
               color=bar_colors, height=0.6)

for bar, val in zip(bars, df_sorted["Performance"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

ax.set_xlabel("Performance score  (P = 0.40·F1 + 0.35·Recall_stress + 0.25·MCC_norm)",
              fontsize=10)
ax.set_xlim(0, df_sorted["Performance"].max() * 1.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.3)

patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
ax.legend(handles=patches, loc="lower right", fontsize=9)
ax.set_title("Performance score across all signal combinations", fontsize=12, pad=12)

plt.tight_layout()
plt.savefig("fig1_performance_ablation.pdf", dpi=300, bbox_inches="tight")
plt.savefig("fig1_performance_ablation.png", dpi=300, bbox_inches="tight")
plt.show()
''', language="python")

    elif chart_choice.startswith("Chart 2"):
        st.subheader("Chart 2 — Lambda sensitivity line chart")
        st.code('''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("optimised_scores.csv")   # your scores file

LAMBDAS = [0.2, 0.4, 0.6, 0.8, 1.0]
HIGHLIGHT = ["ACC", "EDA+ACC", "EDA+ACC+EMG", "All 6 signals", "EDA"]
COLORS    = ["#1D9E75", "#378ADD", "#7F77DD", "#888780", "#D85A30"]

fig, ax = plt.subplots(figsize=(9, 5))

for sig, color in zip(HIGHLIGHT, COLORS):
    row = df[df["Signal"] == sig].iloc[0]
    y = [lam * row["Performance"] + (1-lam)*(1-row["Cost_norm"])
         for lam in LAMBDAS]
    ax.plot(LAMBDAS, y, marker="o", label=sig, color=color,
            linewidth=2, markersize=6)

ax.set_xlabel("λ  (0 = prioritise cost,  1 = prioritise performance)", fontsize=11)
ax.set_ylabel("Optimised score  O = λP + (1−λ)(1−C)", fontsize=11)
ax.set_xlim(0.15, 1.05)
ax.set_ylim(0.3, 0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.2)
ax.legend(fontsize=9, loc="upper left")
ax.set_title("How the winner changes with deployment context (λ)", fontsize=12, pad=12)

# Annotate winner transitions
ax.axvspan(0.2, 0.35, alpha=0.06, color="#1D9E75", label="ACC dominates")
ax.axvspan(0.35, 0.75, alpha=0.06, color="#378ADD")
ax.axvspan(0.75, 1.0,  alpha=0.06, color="#7F77DD")

plt.tight_layout()
plt.savefig("fig2_lambda_sensitivity.pdf", dpi=300, bbox_inches="tight")
plt.show()
''', language="python")

    elif chart_choice.startswith("Chart 3"):
        st.subheader("Chart 3 — Performance vs Cost scatter")
        st.code('''
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("optimised_scores.csv")

TYPE_COLORS = {
    "Unimodal": "#888780",
    "2-signal": "#378ADD",
    "3-signal": "#1D9E75",
    "6-signal": "#7F77DD",
}
colors = [TYPE_COLORS[t] for t in df["Type"]]

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(df["Cost_norm"], df["Performance"],
                c=colors, s=80, alpha=0.85, edgecolors="white", linewidths=0.5)

# Label key points
key_signals = ["ACC", "EDA+ACC", "EDA+ACC+EMG", "All 6 signals", "EMG"]
for _, row in df[df["Signal"].isin(key_signals)].iterrows():
    ax.annotate(row["Signal"],
                xy=(row["Cost_norm"], row["Performance"]),
                xytext=(6, 4), textcoords="offset points", fontsize=8)

ax.set_xlabel("Normalised cost  (higher = more expensive / less practical)", fontsize=11)
ax.set_ylabel("Performance score  P", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.2)
ax.set_title("Performance vs Cost — each dot is a signal combination", fontsize=12, pad=12)

import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
ax.legend(handles=patches, fontsize=9)

plt.tight_layout()
plt.savefig("fig3_performance_vs_cost.pdf", dpi=300, bbox_inches="tight")
plt.show()
''', language="python")

    elif chart_choice.startswith("Chart 4"):
        st.subheader("Chart 4 — Unimodal comparison")
        st.code('''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

signals  = ["HRV", "EDA", "ACC", "Resp", "Temp", "EMG"]
f1       = [0.240, 0.602, 0.560, 0.467, 0.372, 0.413]
recall_s = [0.258, 0.839, 0.774, 0.419, 0.355, 0.194]
perf     = [0.304, 0.735, 0.674, 0.497, 0.411, 0.371]

x  = np.arange(len(signals))
w  = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w,   f1,       w, label="F1-macro",      color="#378ADD", alpha=0.85)
ax.bar(x,       recall_s, w, label="Recall (stress)",color="#1D9E75", alpha=0.85)
ax.bar(x + w,   perf,     w, label="Performance P",  color="#7F77DD", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(signals, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9)
ax.set_title("Unimodal signal comparison — F1, Stress Recall, Performance", fontsize=12, pad=12)

plt.tight_layout()
plt.savefig("fig4_unimodal_comparison.pdf", dpi=300, bbox_inches="tight")
plt.show()
''', language="python")

    elif chart_choice.startswith("Chart 5"):
        st.subheader("Chart 5 — Winner heatmap")
        st.code('''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv("optimised_scores.csv")
lambdas = [0.2, 0.4, 0.6, 0.8, 1.0]

# Build matrix: rows=signals, cols=lambda values
signals = df["Signal"].tolist()
matrix = np.zeros((len(signals), len(lambdas)))

for j, lam in enumerate(lambdas):
    scores = lam * df["Performance"] + (1-lam)*(1-df["Cost_norm"])
    for i, sig in enumerate(signals):
        matrix[i, j] = scores[df["Signal"]==sig].values[0]

heatmap_df = pd.DataFrame(matrix, index=signals,
                           columns=[f"λ={l}" for l in lambdas])
heatmap_df = heatmap_df.sort_values("λ=0.5", ascending=False)

fig, ax = plt.subplots(figsize=(8, 9))
sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGn",
            linewidths=0.3, ax=ax, cbar_kws={"label":"O score"})
ax.set_title("Optimised score across all λ values", fontsize=12, pad=12)
ax.set_xlabel("λ value", fontsize=11)
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("fig5_lambda_heatmap.pdf", dpi=300, bbox_inches="tight")
plt.show()
''', language="python")

    st.markdown("---")
    # st.info("All charts save as both `.pdf` (vector, for paper submission) and `.png` (raster, for presentations). Use the `.pdf` version for IEEE/ACM submissions — it scales without blurring.")