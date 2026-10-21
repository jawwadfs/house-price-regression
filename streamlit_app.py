"""
House Price Regression -- Jury 2 frontend (Streamlit)

Trains all four models at start-up from the bundled Ames CSV using model.py,
the SAME code the Colab notebook uses — so every number shown here is real
and reproducible, not hard-coded.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model import (
    RANDOM_STATE, TARGET, NONE_MEANS_ABSENT, BEST_GB_PARAMS, FINAL_MODEL_KEY,
    AmesFeatureEngineer, build_pipeline, build_jury1_pipeline, remove_outliers,
)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Regression – Jury 2",
    page_icon="🏠",
    layout="wide",
)

# ── load & train (cached so it only runs once per session) ───────────────────
@st.cache_resource(show_spinner="Training models on Ames data… (~30 s)")
def load_and_train():
    df = pd.read_csv("ames_housing.csv")
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    X_clean, y_clean = remove_outliers(X_train, y_train)

    models = {
        "Jury 1 – Linear Regression": build_jury1_pipeline(
            LinearRegression(), num_cols, cat_cols
        ).fit(X_train, y_train),
        "Jury 1 – Gradient Boosting": build_jury1_pipeline(
            GradientBoostingRegressor(
                n_estimators=200, max_depth=3,
                learning_rate=0.1, random_state=RANDOM_STATE
            ),
            num_cols, cat_cols,
        ).fit(X_train, y_train),
        "Jury 2 – Ridge (regularised)": build_pipeline(
            RidgeCV(alphas=np.logspace(-1, 3, 30)), True, True
        ).fit(X_clean, y_clean),
        "Jury 2 – Tuned Gradient Boosting": build_pipeline(
            GradientBoostingRegressor(**BEST_GB_PARAMS), True, True
        ).fit(X_clean, y_clean),
    }
    final_name = {
        "ridge": "Jury 2 – Ridge (regularised)",
        "gb":    "Jury 2 – Tuned Gradient Boosting",
    }[FINAL_MODEL_KEY]
    final = models[final_name]

    # anomaly flag (one Partial-sale test house)
    anomaly = (
        (X_test["Gr Liv Area"] > 4000) &
        (X_test["Sale Condition"] == "Partial")
    ).values

    # performance table
    def _rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    perf_rows = []
    for name, m in models.items():
        p = m.predict(X_test)
        perf_rows.append({
            "Model": name,
            "RMSE – all 586 test houses": f"${_rmse(y_test, p):,.0f}",
            "RMSE – excl. 1 anomaly":     f"${_rmse(y_test[~anomaly], p[~anomaly]):,.0f}",
            "MAE":                         f"${mean_absolute_error(y_test, p):,.0f}",
            "Avg % error":                 f"{np.mean(np.abs(y_test - p) / y_test) * 100:.1f}%",
        })
    perf_df = pd.DataFrame(perf_rows)

    # 80 % prediction band from held-out residuals
    p_final = final.predict(X_test)
    rel = np.log(y_test.values[~anomaly] / p_final[~anomaly])
    q10, q90 = np.quantile(rel, 0.10), np.quantile(rel, 0.90)

    # permutation importance
    pi = permutation_importance(
        final, X_test[~anomaly], y_test[~anomaly],
        scoring="neg_root_mean_squared_error",
        n_repeats=3, random_state=RANDOM_STATE, n_jobs=1,
    )
    imp = (
        pd.Series(pi.importances_mean, index=X_test.columns)
        .sort_values(ascending=False)
        .head(12)
    )

    # typical-house defaults
    cleaned = AmesFeatureEngineer().fit(X_train).transform(X_train)
    default_row = {}
    for c in X.columns:
        if c in num_cols:
            default_row[c] = float(X_train[c].median())
        else:
            mode = cleaned[c].mode()[0]
            default_row[c] = (
                np.nan if (c in NONE_MEANS_ABSENT and mode == "None") else mode
            )

    has_garage = X_train["Garage Cars"].fillna(0) > 0
    area_per_car = float(
        (X_train.loc[has_garage, "Garage Area"] /
         X_train.loc[has_garage, "Garage Cars"]).median()
    )

    return dict(
        X=X, X_train=X_train, X_test=X_test,
        y_test=y_test, anomaly=anomaly,
        models=models, final=final, final_name=final_name,
        perf_df=perf_df, q10=q10, q90=q90, imp=imp,
        default_row=default_row, area_per_car=area_per_car,
        num_cols=num_cols,
    )


state = load_and_train()
X            = state["X"]
X_train      = state["X_train"]
models       = state["models"]
final        = state["final"]
final_name   = state["final_name"]
perf_df      = state["perf_df"]
q10, q90     = state["q10"], state["q90"]
imp          = state["imp"]
default_row  = state["default_row"]
area_per_car = state["area_per_car"]

# ── UI labels & defaults ─────────────────────────────────────────────────────
LABELS = {
    "Neighborhood":    "Neighborhood",
    "Overall Qual":    "Overall quality (1–10)",
    "Overall Cond":    "Overall condition (1–9)",
    "Year Built":      "Year built",
    "Year Remod/Add":  "Year remodelled",
    "1st Flr SF":      "1st floor area (sq ft)",
    "2nd Flr SF":      "2nd floor area (sq ft)",
    "Total Bsmt SF":   "Basement area (sq ft)",
    "Lot Area":        "Lot area (sq ft)",
    "Bedroom AbvGr":   "Bedrooms above grade",
    "Full Bath":       "Full bathrooms",
    "Half Bath":       "Half bathrooms",
    "Garage Cars":     "Garage capacity (cars)",
    "Fireplaces":      "Fireplaces",
    "Kitchen Qual":    "Kitchen quality",
    "Exter Qual":      "Exterior quality",
    "Central Air":     "Central air (Y/N)",
}

D = {
    "Neighborhood":   default_row["Neighborhood"],
    "Overall Qual":   int(default_row["Overall Qual"]),
    "Overall Cond":   int(default_row["Overall Cond"]),
    "Year Built":     int(default_row["Year Built"]),
    "Year Remod/Add": int(default_row["Year Remod/Add"]),
    "1st Flr SF":     int(default_row["1st Flr SF"]),
    "2nd Flr SF":     0,
    "Total Bsmt SF":  int(default_row["Total Bsmt SF"]),
    "Lot Area":       int(default_row["Lot Area"]),
    "Bedroom AbvGr":  int(default_row["Bedroom AbvGr"]),
    "Full Bath":      int(default_row["Full Bath"]),
    "Half Bath":      int(default_row["Half Bath"]),
    "Garage Cars":    int(default_row["Garage Cars"]),
    "Fireplaces":     int(default_row["Fireplaces"]),
    "Kitchen Qual":   default_row["Kitchen Qual"],
    "Exter Qual":     default_row["Exter Qual"],
    "Central Air":    default_row["Central Air"],
}
UI_KEYS = list(D.keys())


def build_row(v):
    """Turn ~17 UI inputs into a full 79-column raw Ames row."""
    r = dict(default_row)
    for k in UI_KEYS:
        r[k] = v[k]
    r["Year Remod/Add"] = max(int(r["Year Remod/Add"]), int(r["Year Built"]))
    r["Gr Liv Area"]    = r["1st Flr SF"] + r["2nd Flr SF"]
    r["Low Qual Fin SF"] = 0
    r["TotRms AbvGrd"]  = r["Bedroom AbvGr"] + 3 + (1 if r["2nd Flr SF"] > 0 else 0)
    b = r["Total Bsmt SF"]
    r["Bsmt Unf SF"] = b; r["BsmtFin SF 1"] = 0; r["BsmtFin SF 2"] = 0
    if b == 0:
        for c in ["Bsmt Qual","Bsmt Cond","Bsmt Exposure","BsmtFin Type 1","BsmtFin Type 2"]:
            r[c] = np.nan
        r["Bsmt Full Bath"] = r["Bsmt Half Bath"] = 0
    r["Garage Area"] = r["Garage Cars"] * area_per_car
    if r["Garage Cars"] == 0:
        for c in ["Garage Type","Garage Finish","Garage Qual","Garage Cond"]:
            r[c] = np.nan
        r["Garage Yr Blt"] = np.nan
    if r["Fireplaces"] == 0:
        r["Fireplace Qu"] = np.nan
    return pd.DataFrame([r])[X.columns]


def predict_all(v):
    row = build_row(v)
    return {n: float(m.predict(row)[0]) for n, m in models.items()}


# ── tabs ──────────────────────────────────────────────────────────────────────
st.title("🏠 House Price Regression – Jury 2")
tab_pred, tab_perf, tab_about = st.tabs(
    ["Predict", "Model performance (Jury 1 vs Jury 2)", "About & roadmap"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown(
        "Set the house characteristics below. Any column you don't set uses the "
        "**median Ames value** as a stand-in."
    )

    nb_choices  = sorted(X["Neighborhood"].unique().tolist())
    q_choices   = [c for c in ["Ex","Gd","TA","Fa","Po"] if c in set(X["Kitchen Qual"].dropna())]
    e_choices   = [c for c in ["Ex","Gd","TA","Fa"]      if c in set(X["Exter Qual"].dropna())]

    col_left, col_right = st.columns(2)

    with col_left:
        v = {}
        v["Neighborhood"]   = st.selectbox(LABELS["Neighborhood"],   nb_choices, index=nb_choices.index(D["Neighborhood"]))
        v["Overall Qual"]   = st.slider(LABELS["Overall Qual"],   1, 10, D["Overall Qual"])
        v["Overall Cond"]   = st.slider(LABELS["Overall Cond"],   1,  9, D["Overall Cond"])
        v["Year Built"]     = st.slider(LABELS["Year Built"],  1880, 2010, D["Year Built"])
        v["Year Remod/Add"] = st.slider(LABELS["Year Remod/Add"], 1950, 2010, D["Year Remod/Add"])
        v["1st Flr SF"]     = st.slider(LABELS["1st Flr SF"],   300, 2500, D["1st Flr SF"], step=10)
        v["2nd Flr SF"]     = st.slider(LABELS["2nd Flr SF"],     0, 1500, D["2nd Flr SF"], step=10)
        v["Total Bsmt SF"]  = st.slider(LABELS["Total Bsmt SF"],   0, 2500, D["Total Bsmt SF"], step=10)
        v["Lot Area"]       = st.slider(LABELS["Lot Area"],    1500, 30000, D["Lot Area"], step=100)

    with col_right:
        v["Bedroom AbvGr"]  = st.slider(LABELS["Bedroom AbvGr"],  0, 6, D["Bedroom AbvGr"])
        v["Full Bath"]      = st.slider(LABELS["Full Bath"],      0, 4, D["Full Bath"])
        v["Half Bath"]      = st.slider(LABELS["Half Bath"],      0, 2, D["Half Bath"])
        v["Garage Cars"]    = st.slider(LABELS["Garage Cars"],    0, 4, D["Garage Cars"])
        v["Fireplaces"]     = st.slider(LABELS["Fireplaces"],     0, 3, D["Fireplaces"])
        v["Kitchen Qual"]   = st.selectbox(LABELS["Kitchen Qual"], q_choices, index=q_choices.index(D["Kitchen Qual"]) if D["Kitchen Qual"] in q_choices else 0)
        v["Exter Qual"]     = st.selectbox(LABELS["Exter Qual"],   e_choices, index=e_choices.index(D["Exter Qual"])   if D["Exter Qual"]   in e_choices else 0)
        v["Central Air"]    = st.selectbox(LABELS["Central Air"],  ["Y","N"], index=0 if D["Central Air"] == "Y" else 1)

    st.divider()

    # ── live prediction (no button needed – Streamlit reruns on every widget change)
    preds = predict_all(v)
    final_pred = preds[final_name]
    lo = final_pred * np.exp(q10)
    hi = final_pred * np.exp(q90)

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted price", f"${final_pred:,.0f}")
    m2.metric("80 % range – low",  f"${lo:,.0f}")
    m3.metric("80 % range – high", f"${hi:,.0f}")
    st.caption(f"Final model: **{final_name}** · Band estimated from held-out residuals")

    st.subheader("All models side by side")
    cmp_df = pd.DataFrame({
        "Model": list(preds.keys()),
        "Predicted price": [f"${p:,.0f}" for p in preds.values()],
        "vs. final model": [f"{p - final_pred:+,.0f}" for p in preds.values()],
    })
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    # ── sensitivity chart
    st.subheader("Why this price?")
    st.caption("Each bar shows how much the price changes when that input is reset to the typical Ames value. Not SHAP — one input changed at a time.")
    deltas = {}
    for k in UI_KEYS:
        v2 = dict(v); v2[k] = D[k]
        deltas[LABELS[k]] = final_pred - float(final.predict(build_row(v2))[0])
    s = pd.Series(deltas)
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(10)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s.index, s.values, color=["#2E7D32" if x >= 0 else "#B4530A" for x in s.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Effect on price vs. a typical Ames house ($)")
    ax.set_title("Why this price? (one input changed at a time)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown(
        "Evaluated on the same **586 held-out test houses** (80/20 split, seed 42). "
        "One test house is a documented anomaly — a 4,676 sq ft 'Partial' sale at "
        "\\$184,750 — that dominates RMSE for linear models. Both views are shown."
    )
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.subheader(f"Permutation importance – {final_name}")
    st.caption("How much does RMSE increase when each input column is randomly shuffled? Works for any model, unlike tree-based importance.")
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    s2 = imp[::-1]
    ax2.barh(s2.index, s2.values, color="#1E2761")
    ax2.set_xlabel("Increase in RMSE ($) when this feature is shuffled")
    ax2.set_title(f"Permutation importance – {final_name}")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### How it works
`Inputs → typical-house defaults for the other ~60 columns → feature engineering (leak-safe) → model → price`

### What changed since Jury 1

| Jury 1 weakness | Jury 2 fix |
|---|---|
| Blank values treated as lost data | Domain-aware cleaning: blank = feature absent |
| Raw columns only | 23 engineered features (total area, bathrooms, house age, quality × size) |
| Errors grew with price | Log-transformed target |
| One "Partial sale" anomaly distorted training | Outlier removal in training only; results reported with and without it |
| One split, no tuning | 5-fold cross-validation and RandomizedSearchCV |
| Unstable coefficients on rare categories | Rare-category grouping, Ridge and Lasso |
| Only tree-based importance | Permutation importance (works for any model) |

### Honest limitations
- Trained on Ames, Iowa (2006–2010) sales only — not valid for other cities or years
- One 'Partial sale' anomaly dominates full-test RMSE; both views are reported
- Explanations here are simple one-at-a-time sensitivities, **not SHAP**
- Prices beyond ~4,000 sq ft are poorly covered by the data
- Absolute errors still grow with price; percentage error is flat

### Roadmap – Jury 3 (not built yet)
A GenAI explanation layer will turn the prediction + feature effects into a
plain-English explanation and answer follow-up questions.
**The regression model stays the predictor; GenAI only explains.**
""")
