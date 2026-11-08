"""
ShopSense - Purchase Intent Classifier (ML Assignment 2, BITS WILP M.Tech AIML)
Loads the 5 pre-trained pipelines from model/ and evaluates them on an
uploaded test CSV. No model is retrained inside the app.
"""
import io
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
NUMERIC = FEATURES[:10]
SAMPLE_PATH = BASE / "test_data.csv"
MODEL_COLORS = {"Logistic Regression": "#1d4ed8", "Decision Tree": "#0f766e", "KNN": "#9333ea",
                "Naive Bayes": "#ea580c", "Support Vector Machine": "#dc2626"}
MODEL_NOTES = {
    "Logistic Regression": "Linear model · L2, C=1.0 · log1p + scaling + one-hot",
    "Decision Tree": "Tuned by 5-fold GridSearchCV · max_depth=8, min_samples_leaf=50",
    "KNN": "Tuned by 5-fold GridSearchCV · k=31, uniform weights, Euclidean",
    "Naive Bayes": "GaussianNB · var_smoothing=1e-2",
    "Support Vector Machine": "RBF kernel · C=1.0, gamma='scale' · Platt-scaled probabilities",
}

st.set_page_config(page_title="ShopSense | Purchase Intent", page_icon="🛒", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {background: linear-gradient(90deg,#0f766e,#1d4ed8); color:#fff;
       padding:18px 24px; border-radius:12px; margin-bottom:14px;}
.hero h1 {margin:0; font-size:1.9rem; color:#fff;}
.hero p {margin:4px 0 0 0; opacity:.92;}
div[data-testid="stMetric"] {background:#ffffff; border:1px solid #99d5cf; border-left:5px solid #0f766e;
       border-radius:10px; padding:10px 14px; box-shadow:0 1px 3px rgba(0,0,0,.08);}
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {color:#475569 !important; font-weight:600;}
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {color:#0f172a !important;}
</style>
<div class="hero">
  <h1>🛒 ShopSense — Will this visitor buy?</h1>
  <p>Online Shoppers Purchasing Intention (UCI) · 5 pre-trained classifiers · evaluated on the test data you upload</p>
</div>
""", unsafe_allow_html=True)


# ---------------- helpers ----------------
@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in FILES.items()}


def to_binary(series):
    s = series.astype(str).str.strip().str.upper()
    mapping = {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "YES": 1, "NO": 0}
    out = s.map(mapping)
    if out.isna().any():
        bad = sorted(s[out.isna()].unique())[:5]
        raise ValueError(f"Column '{TARGET}' has values I can't read as TRUE/FALSE: {bad}")
    return out.astype(int)


def prepare(df):
    """Validate the upload. Returns X, y (None if no target), list of warnings."""
    notes = []
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required feature columns: {missing}")
    extra = [c for c in df.columns if c not in FEATURES + [TARGET]]
    if extra:
        notes.append(f"Ignored {len(extra)} extra column(s): {extra[:5]}")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=FEATURES + ([TARGET] if TARGET in df.columns else []))
    if len(df) < n_before:
        notes.append(f"Dropped {n_before - len(df)} row(s) with missing / non-numeric values.")
    if df.empty:
        raise ValueError("No usable rows left after removing rows with missing values.")
    X = df[FEATURES].copy()
    X["Weekend"] = X["Weekend"].astype(str).str.strip().str.upper()
    y = to_binary(df[TARGET]) if TARGET in df.columns else None
    return X.reset_index(drop=True), (None if y is None else y.reset_index(drop=True)), notes


def compute_metrics(y, pred, proba):
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba) if y.nunique() > 1 else np.nan,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred) if y.nunique() > 1 else np.nan,
    }


@st.cache_data(show_spinner="Scoring test data with the saved models…")
def all_probabilities(csv_bytes):
    """Purchase probabilities of every saved model for the uploaded file (cached per file)."""
    X, y, notes = prepare(pd.read_csv(io.BytesIO(csv_bytes)))
    models = load_models()
    probas = {name: m.predict_proba(X)[:, 1] for name, m in models.items()}
    # native predictions (used at the default 0.50 threshold so results match the README exactly;
    # for SVM, predict() uses the decision function, not the Platt-scaled probability)
    preds = {name: m.predict(X).astype(int) for name, m in models.items()}
    return X, y, notes, probas, preds


def predict_at(name, threshold):
    if abs(threshold - 0.5) < 1e-9:
        return preds[name]
    return (probas[name] >= threshold).astype(int)


def fmt(v):
    return "n/a" if pd.isna(v) else f"{v:.4f}"


# ---------------- sidebar ----------------
models = load_models()
with st.sidebar:
    st.header("⚙️ Controls")
    model_name = st.selectbox("Choose a model", list(models.keys()), index=1)
    st.caption(MODEL_NOTES[model_name])
    threshold = st.slider("Decision threshold (P(buy) ≥ threshold → buyer)", 0.05, 0.95, 0.50, 0.05,
                          help="0.50 is the default used in the README. Lower it to catch more buyers "
                               "(higher recall, lower precision).")
    st.divider()
    st.subheader("📂 Test data")
    source = st.radio("Data source", ["Upload my CSV", "Use bundled test_data.csv"], index=1,
                      help="The bundled held-out test set loads by default; switch to upload your own CSV.")
    uploaded = None
    if source == "Upload my CSV":
        uploaded = st.file_uploader("CSV with the 17 features + Revenue", type=["csv"])
    if SAMPLE_PATH.exists():
        st.download_button("⬇️ Download sample test_data.csv", SAMPLE_PATH.read_bytes(),
                           file_name="test_data.csv", mime="text/csv", use_container_width=True)
    with st.expander("Required columns"):
        st.code(", ".join(FEATURES + [TARGET]), language=None)

if source == "Use bundled test_data.csv":
    csv_bytes, source_name = SAMPLE_PATH.read_bytes(), "bundled test_data.csv"
elif uploaded is not None:
    csv_bytes, source_name = uploaded.getvalue(), uploaded.name
else:
    st.info("👈 Upload a test CSV in the sidebar, or switch the data source to the bundled "
            "**test_data.csv** (you can also download it there and upload it back).")
    st.stop()

try:
    X, y, notes, probas, preds = all_probabilities(csv_bytes)
except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
    st.error(f"Could not use **{source_name}**: {e}")
    st.stop()

for n in notes:
    st.warning(n)

has_target = y is not None
proba = probas[model_name]
pred = predict_at(model_name, threshold)

summary = f"Data source: **{source_name}** · {len(X):,} rows used"
if has_target:
    summary += f" · actual buyers: {y.mean():.1%}"
st.caption(summary)
if not has_target:
    st.warning(f"No '{TARGET}' column found — showing predictions only. Add the target column to see "
               "evaluation metrics and the confusion matrix.")

tabs = st.tabs(["📊 Metrics", "🧮 Confusion matrix & report", "🏁 Compare all models",
                "🔎 Predictions", "🗂️ Data overview"])

# ---- Metrics ----
with tabs[0]:
    st.subheader(f"{model_name} — evaluation metrics (threshold {threshold:.2f})")
    if has_target:
        metrics = compute_metrics(y, pred, proba)
        for col, (k, v) in zip(st.columns(6), metrics.items()):
            col.metric(k, fmt(v))
        if y.nunique() < 2:
            st.info("The data contains only one class, so AUC and MCC are undefined (n/a).")
        else:
            c1, c2 = st.columns(2)
            fpr, tpr, _ = roc_curve(y, proba)
            fig, ax = plt.subplots(figsize=(5, 3.8))
            ax.plot(fpr, tpr, color=MODEL_COLORS[model_name], lw=2, label=f"AUC = {metrics['AUC']:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title("ROC curve"); ax.legend(loc="lower right")
            c1.pyplot(fig); plt.close(fig)
            fig, ax = plt.subplots(figsize=(5, 3.8))
            ax.hist(proba[y == 0], bins=30, alpha=.6, label="Actual: no purchase", color="#94a3b8")
            ax.hist(proba[y == 1], bins=30, alpha=.7, label="Actual: purchase", color=MODEL_COLORS[model_name])
            ax.axvline(threshold, color="black", ls="--", lw=1, label="threshold")
            ax.set_xlabel("Predicted P(purchase)"); ax.set_ylabel("Sessions"); ax.set_yscale("log")
            ax.set_title("How well are the classes separated?"); ax.legend(fontsize=8)
            c2.pyplot(fig); plt.close(fig)
    else:
        c1, c2 = st.columns(2)
        c1.metric("Sessions predicted to buy", f"{pred.sum():,}")
        c2.metric("Share predicted to buy", f"{pred.mean():.1%}")

# ---- Confusion matrix ----
with tabs[1]:
    if not has_target:
        st.info("Needs the Revenue column.")
    else:
        left, right = st.columns([1, 1.2])
        normalise = left.toggle("Show as % of actual class", value=False)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalise else cm
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        sns.heatmap(shown, annot=True, fmt=".1%" if normalise else "d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No purchase", "Purchase"], yticklabels=["No purchase", "Purchase"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix — {model_name}")
        left.pyplot(fig); plt.close(fig)
        rep = classification_report(y, pred, labels=[0, 1], target_names=["No purchase", "Purchase"],
                                    output_dict=True, zero_division=0)
        right.markdown("**Classification report**")
        right.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        right.markdown(f"- ✅ Buyers correctly flagged (TP): **{tp}**  \n- ❌ Buyers missed (FN): **{fn}**  \n"
                       f"- ⚠️ False alarms (FP): **{fp}**  \n- ✅ Non-buyers correctly ignored (TN): **{tn}**")

# ---- Compare ----
with tabs[2]:
    if not has_target:
        st.info("Needs the Revenue column.")
    else:
        st.subheader(f"All 5 models on this data (threshold {threshold:.2f})")
        rows = {n: compute_metrics(y, predict_at(n, threshold), p) for n, p in probas.items()}
        comp = pd.DataFrame(rows).T.round(4)
        comp.index.name = "ML Model Name"
        st.dataframe(comp.style.highlight_max(axis=0, color="#bbf7d0").format("{:.4f}", na_rep="n/a"),
                     use_container_width=True)
        st.download_button("⬇️ Download comparison table", comp.to_csv().encode(),
                           file_name="model_comparison.csv")
        c1, c2 = st.columns(2)
        metric_pick = c1.radio("Bar chart metric", list(comp.columns), horizontal=True, index=5)
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        order = comp[metric_pick].fillna(0).sort_values()
        ax.barh(order.index, order.values, color=[MODEL_COLORS[n] for n in order.index])
        for i, v in enumerate(order.values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.12); ax.set_xlabel(metric_pick)
        c1.pyplot(fig); plt.close(fig)
        if y.nunique() > 1:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for n, p in probas.items():
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, lw=1.6, color=MODEL_COLORS[n], label=f"{n} ({roc_auc_score(y, p):.3f})")
            ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title("ROC curves — all models"); ax.legend(fontsize=7, loc="lower right")
            c2.pyplot(fig); plt.close(fig)
            best = comp["MCC"].idxmax()
            st.success(f"🏆 Best model by MCC on this data: **{best}** (MCC = {comp.loc[best, 'MCC']:.4f})")

# ---- Predictions ----
with tabs[3]:
    out = X.copy()
    if has_target:
        out[TARGET] = np.where(y == 1, "TRUE", "FALSE")
    out["Predicted_Revenue"] = np.where(pred == 1, "TRUE", "FALSE")
    out["Purchase_Probability"] = proba.round(4)
    only_buyers = st.checkbox("Show only sessions predicted to buy")
    view = out[out["Predicted_Revenue"] == "TRUE"] if only_buyers else out
    st.dataframe(view.head(500), use_container_width=True)
    st.caption(f"Showing {min(len(view), 500):,} of {len(view):,} rows.")
    st.download_button("⬇️ Download all predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv")

# ---- Data overview ----
with tabs[4]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(X):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Buyers (actual)", f"{y.mean():.1%}" if has_target else "n/a")
    st.markdown("**Numeric feature summary**")
    st.dataframe(X[NUMERIC].describe().T.round(2), use_container_width=True)
    if has_target:
        fig, ax = plt.subplots(figsize=(6, 3))
        rate = pd.DataFrame({"Month": X["Month"], "buy": y}).groupby("Month")["buy"].mean()
        order = [m for m in ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if m in rate.index]
        rate = rate.reindex(order + [m for m in rate.index if m not in order])
        ax.bar(rate.index, rate.values, color="#0f766e")
        ax.set_ylabel("Purchase rate"); ax.set_title("Actual purchase rate by month")
        st.pyplot(fig); plt.close(fig)

st.caption("Models were trained offline (train_models.py) and are only loaded here with joblib — "
           "no retraining on uploaded data.")
