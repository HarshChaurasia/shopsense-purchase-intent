# 🛒 ShopSense — Predicting Online Purchase Intent with 5 Classifiers

**ML Assignment 2 · AIMLZG565 Machine Learning · BITS WILP M.Tech (AIML)**
**Author:** Harsh

- **GitHub repository:** __REPO_URL__
- **Live Streamlit app:** __APP_URL__

---

## a. Problem statement

An e-commerce site records how each visitor browses during a session: which kinds of pages they open, how long they stay, bounce and exit rates, and the "page value" of the pages they view. The question is **whether a session will end in a purchase** (`Revenue = TRUE`), using only this session behaviour.

This is a **binary classification** problem. It is also **imbalanced**: only about 15.6% of sessions buy. Accuracy alone is therefore misleading, so the models are compared with AUC, Precision, Recall, F1 and the **Matthews Correlation Coefficient (MCC)**. MCC is the most balanced single score for imbalanced binary data.

Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same split and compared. They are served through an interactive Streamlit app that evaluates the saved models on uploaded test data **without retraining**.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | **Online Shoppers Purchasing Intention Dataset** |
| Source | UCI Machine Learning Repository (id 468), Sakar et al., 2018 |
| Instances | 12,330 sessions (12,205 after removing 125 exact duplicates) |
| Features | **17** input features (10 numerical + 7 categorical) |
| Target | `Revenue` (TRUE = purchase, FALSE = no purchase) |
| Class balance | 84.4% FALSE / 15.6% TRUE |
| Missing values | None |

**Features**

| Feature | Type | Meaning |
|---|---|---|
| Administrative, Informational, ProductRelated | numeric (count) | Number of pages of each type visited in the session |
| Administrative_Duration, Informational_Duration, ProductRelated_Duration | numeric (seconds) | Time spent on each page type |
| BounceRates | numeric | Avg. % of visitors who entered on these pages and left without another request |
| ExitRates | numeric | Avg. % of page views that were the last in a session |
| PageValues | numeric | Avg. value of the pages visited before a transaction (Google Analytics) |
| SpecialDay | numeric (0–1) | Closeness of the visit to a special day (e.g. Mother's Day, Valentine's Day) |
| Month | categorical | Month of the visit |
| OperatingSystems, Browser, Region, TrafficType | categorical (integer codes) | Technical and acquisition details of the visitor |
| VisitorType | categorical | Returning_Visitor / New_Visitor / Other |
| Weekend | boolean | Whether the visit was on a weekend |

**Pre-processing** (inside each saved scikit-learn `Pipeline`, so the app applies exactly the same steps):
- Removed duplicate rows. Stratified **80/20 train/test split** (`random_state = 7`): 9,764 training rows and **2,441 test rows** (`test_data.csv`).
- Numerical features: `log1p` (the counts and durations are heavily right-skewed), then `StandardScaler`. The Decision Tree uses only scaling, since trees don't need it.
- Categorical features, including the integer-coded OS/Browser/Region/TrafficType: `OneHotEncoder(handle_unknown="ignore")`.

## c. GitHub repository

__REPO_URL__

```
├── app.py                          # Streamlit app (loads saved models, no retraining)
├── train_models.py                 # training + evaluation script (all 5 models)
├── ML_Assignment2_Training.ipynb   # notebook run on BITS Virtual Lab
├── requirements.txt
├── README.md
├── test_data.csv                   # held-out test set (2,441 rows, 17 features + Revenue)
├── data/online_shoppers_intention.csv   # full original dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics_summary.csv
    └── model_info.json
```

**Reproduce:** `pip install -r requirements.txt` → `python train_models.py` → `streamlit run app.py`

## d. Models used

| Model | Configuration |
|---|---|
| Logistic Regression | L2, C = 1.0, lbfgs, max_iter = 2000 |
| Decision Tree | GridSearchCV (5-fold, F1) over max_depth and min_samples_leaf → **max_depth = 8, min_samples_leaf = 50** |
| KNN | GridSearchCV (5-fold, F1) over k and weights → **k = 31, uniform** |
| Naive Bayes | GaussianNB, var_smoothing = 1e-2 |
| Support Vector Machine | RBF kernel, C = 1.0, gamma = "scale", probability = True |

### Comparison table (held-out test set, 2,441 rows, threshold 0.5)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8984 | 0.9208 | 0.7279 | 0.5602 | 0.6331 | 0.5820 |
| Decision Tree | **0.9050** | **0.9279** | **0.7551** | 0.5812 | **0.6568** | **0.6097** |
| KNN | 0.8968 | 0.9259 | 0.7390 | 0.5262 | 0.6147 | 0.5677 |
| Naive Bayes | 0.7095 | 0.8562 | 0.3367 | **0.8822** | 0.4873 | 0.4135 |
| Support Vector Machine | 0.9041 | 0.9052 | 0.7517 | 0.5785 | 0.6538 | 0.6062 |

*Baseline: predicting "no purchase" for every session gives 84.4% accuracy but an MCC of 0, so only scores well above 0.844 accuracy are meaningful.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong, well-calibrated linear baseline (AUC 0.921, MCC 0.582). **PageValues** has by far the largest positive coefficient, and February / some rare OS codes push predictions down. Precision is good (0.73) but recall is only 0.56: at the default 0.5 threshold it misses ~44% of buyers, because the minority class sits close to the linear boundary. Lowering the threshold in the app trades precision for recall. |
| Decision Tree | **Best overall** (Accuracy 0.905, AUC 0.928, F1 0.657, MCC 0.610). Tuning was essential: depth 8 with at least 50 samples per leaf prevents the overfitting a fully grown tree shows. The tree takes 74.5% of its split importance from **PageValues**, followed by BounceRates and Month = Nov. It captures the threshold-style rule "a session that reaches valuable pages, in a busy shopping month, buys", which a linear model can only approximate. It is also the most interpretable model. |
| KNN | Good ranking ability (AUC 0.926, 2nd best) but the **lowest recall among the strong models (0.526)**. With k = 31 the neighbourhood vote is dominated by the 84% majority class, so borderline buyers get outvoted. Distance-based learning also suffers from the 74-dimensional (one-hot expanded) feature space, and the model is the largest file (it stores the whole training set) and the slowest to predict. |
| Naive Bayes | Clearly the weakest on balanced metrics (Accuracy 0.71, MCC 0.41), but it has the **highest recall (0.88)**: it flags almost every buyer at the cost of many false alarms (precision 0.34). This comes from its independence assumption, which is violated here (page counts and their durations are strongly correlated), and from forcing one-hot and skewed features into Gaussian likelihoods. That makes its probabilities over-confident. It is useful only where missing a buyer is far costlier than a false alarm. |
| Support Vector Machine | Second best on threshold metrics (Accuracy 0.904, F1 0.654, MCC 0.606), almost tied with the Decision Tree. The RBF kernel models non-linear interactions well. However, its **AUC (0.905) is the lowest of the strong models** because Platt-scaled probabilities rank sessions less smoothly than LR/DT. It is also the slowest to train (probability=True needs internal cross-validation). |
| **Overall winner for this dataset** | **Decision Tree** (tuned): it ranks first on Accuracy, AUC, Precision, F1 and MCC, and it is fast and explainable. The SVM is a close runner-up. All models are limited mainly by the class imbalance (recall ≈ 0.53–0.58 at threshold 0.5). In practice the decision threshold or class weights should be tuned to the business cost of a missed buyer. |

## Streamlit app features

1. **CSV upload** of test data (features + `Revenue`), with a one-click download of the bundled `test_data.csv`. Missing columns are validated and reported.
2. **Model selection dropdown** for the 5 saved models, plus an adjustable **decision-threshold slider**.
3. **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC, plus a ROC curve.
4. **Confusion matrix** heatmap and the **classification report**.
5. **Compare all models** tab: a side-by-side table and bar chart, highlighting the best model by MCC.
6. **Predictions** tab with per-row purchase probability and CSV download.

The app only calls `joblib.load` and then `predict` / `predict_proba`. **No model is retrained on the uploaded data.**
