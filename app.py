"""
ShopSense - Purchase Intent Classifier (ML Assignment 2, BITS WILP M.Tech AIML)
Loads the 5 pre-trained pipelines from model/ and evaluates them on an
uploaded test CSV. No model is retrained inside the app.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
INFO = json.loads((MODEL_DIR / "model_info.json").read_text())
FEATURES, TARGET, FILES = INFO["features"], INFO["target"], INFO["files"]

st.set_page_config(page_title="ShopSense | Purchase Intent", page_icon="🛒", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {background: linear-gradient(90deg,#0f766e,#1d4ed8); color:#fff;
       padding:18px 24px; border-radius:12px; margin-bottom:14px;}
.hero h1 {margin:0; font-size:1.9rem;}
.hero p {margin:4px 0 0 0; opacity:.9;}
div[data-testid="stMetric"] {background:rgba(15,118,110,.08); border:1px solid rgba(15,118,110,.25); border-radius:10px; padding:10px 14px;}
</style>
<div class="hero">
  <h1>🛒 ShopSense — Will this visitor buy?</h1>
  <p>Online Shoppers Purchasing Intention (UCI) · 5 classifiers · evaluation on your uploaded test data</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in FILES.items()}


def to_binary(series):
    s = series.astype(str).str.strip().str.upper()
    mapping = {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "YES": 1, "NO": 0}
    out = s.map(mapping)
    if out.isna().any():
        raise ValueError(f"Unrecognised values in '{TARGET}': {sorted(s[out.isna()].unique())[:5]}")
    return out.astype(int)


def prepare(df):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing columns: {missing}")
    X = df[FEATURES].copy()
    X["Weekend"] = X["Weekend"].astype(str).str.upper()
    return X, to_binary(df[TARGET])


def score(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba) if y.nunique() > 1 else np.nan,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred, proba


models = load_models()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    model_name = st.selectbox("Choose a model", list(models.keys()), index=1)
    threshold = st.slider("Decision threshold (positive = buys)", 0.05, 0.95, 0.50, 0.05,
                          help="Default 0.50. Lower it to catch more buyers (higher recall).")
    st.divider()
    st.subheader("📂 Test data")
    uploaded = st.file_uploader("Upload test CSV (features + Revenue)", type=["csv"])
    sample_path = BASE / "test_data.csv"
    if sample_path.exists():
        st.download_button("⬇️ Download sample test_data.csv", sample_path.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    use_sample = st.checkbox("Use bundled test_data.csv instead", value=uploaded is None)

if uploaded is not None and not use_sample:
    data = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample and sample_path.exists():
    data = pd.read_csv(sample_path)
    source = "bundled test_data.csv"
else:
    st.info("👈 Upload a test CSV or tick *Use bundled test_data.csv* to start.")
    st.stop()

try:
    X, y = prepare(data)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.caption(f"Data source: **{source}** · {len(X):,} rows · buyers in data: {y.mean():.1%}")

model = models[model_name]
metrics, _, proba = score(model, X, y)
pred = (proba >= threshold).astype(int)
if threshold != 0.5:  # recompute threshold-dependent metrics
    metrics.update({
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    })

tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "🧮 Confusion matrix & report",
                                  "🏁 Compare all models", "🔎 Predictions"])

with tab1:
    st.subheader(f"{model_name} — evaluation metrics")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(fpr, tpr, color="#1d4ed8", lw=2, label=f"AUC = {metrics['AUC']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    c1, _ = st.columns([1, 1])
    c1.pyplot(fig)

with tab2:
    left, right = st.columns([1, 1.2])
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No purchase", "Purchase"], yticklabels=["No purchase", "Purchase"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix — {model_name}")
    left.pyplot(fig)
    rep = classification_report(y, pred, labels=[0, 1], target_names=["No purchase", "Purchase"],
                                output_dict=True, zero_division=0)
    right.markdown("**Classification report**")
    right.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)
    tn, fp, fn, tp = cm.ravel()
    right.markdown(f"- Correctly flagged buyers (TP): **{tp}**  \n- Missed buyers (FN): **{fn}**  \n"
                   f"- False alarms (FP): **{fp}**  \n- Correct non-buyers (TN): **{tn}**")

with tab3:
    st.subheader("All 5 models on this test data (threshold 0.50)")
    rows = {name: score(m, X, y)[0] for name, m in models.items()}
    comp = pd.DataFrame(rows).T.round(4)
    comp.index.name = "ML Model Name"
    st.dataframe(comp.style.highlight_max(axis=0, color="#bbf7d0"), use_container_width=True)
    metric_pick = st.radio("Plot metric", list(comp.columns), horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(7, 3))
    order = comp[metric_pick].sort_values()
    ax.barh(order.index, order.values, color="#0f766e")
    for i, v in enumerate(order.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.08); ax.set_xlabel(metric_pick)
    st.pyplot(fig)
    best = comp["MCC"].idxmax()
    st.success(f"Best model by MCC on this data: **{best}** (MCC = {comp.loc[best, 'MCC']:.4f})")

with tab4:
    out = data.copy()
    out["Predicted_Revenue"] = np.where(pred == 1, "TRUE", "FALSE")
    out["Purchase_Probability"] = proba.round(4)
    st.dataframe(out.head(200), use_container_width=True)
    st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv")

st.caption("Models were trained offline (train_models.py) and are only loaded here — no retraining on uploaded data.")
