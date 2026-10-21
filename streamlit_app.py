"""
House Price Regression -- Jury 2 frontend (Streamlit)
+ Jury 3 preview: GenAI explanation layer powered by Gemini

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

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Regression – Jury 2",
    page_icon="🏠",
    layout="wide",
)

# ── custom CSS for a cleaner look ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #f0f4ff;
    border: 1px solid #d0d9f0;
    border-radius: 10px;
    padding: 12px 18px;
}
[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #1a3a6b; }
[data-testid="stMetricLabel"] { font-size: 0.85rem !important; color: #555; }
.price-hero {
    background: linear-gradient(135deg, #1a3a6b 0%, #2563eb 100%);
    color: white;
    border-radius: 14px;
    padding: 24px 32px;
    margin-bottom: 18px;
    text-align: center;
}
.price-hero h1 { font-size: 3rem; margin: 0; font-weight: 800; }
.price-hero p  { font-size: 1rem; margin: 6px 0 0; opacity: 0.85; }
.ai-badge {
    display: inline-block;
    background: #7c3aed;
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    vertical-align: middle;
    margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── load & train (cached — runs once per session) ─────────────────────────────
@st.cache_resource(show_spinner="⚙️  Training models on Ames data… (~30 s)")
def load_and_train():
    df = pd.read_csv("ames_housing.csv")
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols  = [c for c in X.columns if c not in num_cols]
    X_clean, y_clean = remove_outliers(X_train, y_train)

    models = {
        "Jury 1 – Linear Regression": build_jury1_pipeline(
            LinearRegression(), num_cols, cat_cols
        ).fit(X_train, y_train),
        "Jury 1 – Gradient Boosting": build_jury1_pipeline(
            GradientBoostingRegressor(
                n_estimators=200, max_depth=3,
                learning_rate=0.1, random_state=RANDOM_STATE,
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

    def _rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    perf_rows = []
    for name, m in models.items():
        p = m.predict(X_test)
        perf_rows.append({
            "Model":                          name,
            "RMSE – all 586 test houses":    f"${_rmse(y_test, p):,.0f}",
            "RMSE – excl. 1 anomaly":        f"${_rmse(y_test[~anomaly], p[~anomaly]):,.0f}",
            "MAE":                            f"${mean_absolute_error(y_test, p):,.0f}",
            "Avg % error":                    f"{np.mean(np.abs(y_test - p) / y_test) * 100:.1f}%",
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

    has_garage  = X_train["Garage Cars"].fillna(0) > 0
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


state        = load_and_train()
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

# ── UI labels & defaults ──────────────────────────────────────────────────────
LABELS = {
    "Neighborhood":   "Neighborhood",
    "Overall Qual":   "Overall quality (1–10)",
    "Overall Cond":   "Overall condition (1–9)",
    "Year Built":     "Year built",
    "Year Remod/Add": "Year remodelled",
    "1st Flr SF":     "1st floor area (sq ft)",
    "2nd Flr SF":     "2nd floor area (sq ft)",
    "Total Bsmt SF":  "Basement area (sq ft)",
    "Lot Area":       "Lot area (sq ft)",
    "Bedroom AbvGr":  "Bedrooms above grade",
    "Full Bath":      "Full bathrooms",
    "Half Bath":      "Half bathrooms",
    "Garage Cars":    "Garage capacity (cars)",
    "Fireplaces":     "Fireplaces",
    "Kitchen Qual":   "Kitchen quality",
    "Exter Qual":     "Exterior quality",
    "Central Air":    "Central air (Y/N)",
}

QUAL_FULL = {"Ex": "Excellent", "Gd": "Good", "TA": "Typical/Average",
             "Fa": "Fair", "Po": "Poor"}

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
    r["Year Remod/Add"]  = max(int(r["Year Remod/Add"]), int(r["Year Built"]))
    r["Gr Liv Area"]     = r["1st Flr SF"] + r["2nd Flr SF"]
    r["Low Qual Fin SF"] = 0
    r["TotRms AbvGrd"]   = r["Bedroom AbvGr"] + 3 + (1 if r["2nd Flr SF"] > 0 else 0)
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


def compute_deltas(v, final_pred):
    """One-at-a-time sensitivity: how much does each input contribute?"""
    deltas = {}
    for k in UI_KEYS:
        v2 = dict(v); v2[k] = D[k]
        deltas[LABELS[k]] = final_pred - float(final.predict(build_row(v2))[0])
    s = pd.Series(deltas)
    return s.reindex(s.abs().sort_values(ascending=False).index).head(10)


# ── Gemini GenAI helper ───────────────────────────────────────────────────────
def _gemini_available():
    try:
        return bool(st.secrets.get("GEMINI_API_KEY", ""))
    except Exception:
        return False


def build_context(v, preds, final_pred, lo, hi, deltas):
    """Build a plain-text context block for the Gemini prompt."""
    top_factors = "\n".join(
        f"  - {label}: {'adds' if delta >= 0 else 'removes'} ${abs(delta):,.0f}"
        for label, delta in deltas.items()
        if abs(delta) > 500
    )
    qual_map = {1:"Very Poor",2:"Poor",3:"Below Average",4:"Below Average",
                5:"Average",6:"Above Average",7:"Good",8:"Very Good",
                9:"Excellent",10:"Excellent"}
    lines = [
        f"House characteristics entered by the user:",
        f"  Neighborhood     : {v['Neighborhood']}",
        f"  Overall quality  : {v['Overall Qual']}/10 ({qual_map.get(v['Overall Qual'], '')})",
        f"  Overall condition: {v['Overall Cond']}/9",
        f"  Year built       : {v['Year Built']}",
        f"  Year remodelled  : {v['Year Remod/Add']}",
        f"  Living area      : {v['1st Flr SF'] + v['2nd Flr SF']:,} sq ft "
        f"  (1st floor {v['1st Flr SF']:,} + 2nd floor {v['2nd Flr SF']:,})",
        f"  Basement area    : {v['Total Bsmt SF']:,} sq ft",
        f"  Lot area         : {v['Lot Area']:,} sq ft",
        f"  Bedrooms         : {v['Bedroom AbvGr']}",
        f"  Bathrooms        : {v['Full Bath']} full + {v['Half Bath']} half",
        f"  Garage           : {v['Garage Cars']}-car",
        f"  Fireplaces       : {v['Fireplaces']}",
        f"  Kitchen quality  : {QUAL_FULL.get(v['Kitchen Qual'], v['Kitchen Qual'])}",
        f"  Exterior quality : {QUAL_FULL.get(v['Exter Qual'], v['Exter Qual'])}",
        f"  Central air      : {'Yes' if v['Central Air'] == 'Y' else 'No'}",
        f"",
        f"Model predictions (all trained on Ames, Iowa 2006-2010 data):",
        *[f"  {n}: ${p:,.0f}" for n, p in preds.items()],
        f"",
        f"Final model ({final_name}): ${final_pred:,.0f}",
        f"80% confidence range: ${lo:,.0f} – ${hi:,.0f}",
        f"",
        f"Top factors vs a typical Ames house (one-at-a-time sensitivity):",
        top_factors,
    ]
    return "\n".join(lines)


def ask_gemini(system_prompt, history, user_message):
    """Send a message to Gemini and return the text response."""
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_prompt,
    )
    chat = model.start_chat(history=history)
    response = chat.send_message(user_message)
    return response.text


SYSTEM_PROMPT = """You are a friendly and knowledgeable real estate pricing assistant.
You help users understand why a machine learning model predicted a specific sale price
for a house in Ames, Iowa (based on 2006–2010 historical data).

Guidelines:
- Be conversational and clear — no jargon.
- Always anchor your explanation to the specific numbers provided.
- Acknowledge model limitations honestly: the model only knows Ames, Iowa 2006-2010 data.
- When listing factors, be specific about dollar amounts from the sensitivity analysis.
- Never invent numbers not provided in the context.
- Keep initial explanations to 3-4 short paragraphs. Follow-up answers can be shorter.
- Do not use markdown headers. Use bullet points sparingly.
"""

# ── session state init ────────────────────────────────────────────────────────
if "chat_history"  not in st.session_state:
    st.session_state.chat_history  = []   # [{role, parts}] for Gemini
if "display_msgs"  not in st.session_state:
    st.session_state.display_msgs  = []   # [{role, content}] for st.chat_message
if "last_context"  not in st.session_state:
    st.session_state.last_context  = ""

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='margin-bottom:0'>🏠 House Price Regression</h2>"
    "<p style='color:#555;margin-top:2px'>Jury 2 · Ames, Iowa · 2006–2010 data</p>",
    unsafe_allow_html=True,
)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_pred, tab_ai, tab_perf, tab_about = st.tabs([
    "🔮 Predict",
    "🤖 AI Explanation",
    "📊 Model Performance",
    "ℹ️ About & Roadmap",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab_pred:
    nb_choices = sorted(X["Neighborhood"].unique().tolist())
    q_choices  = [c for c in ["Ex","Gd","TA","Fa","Po"] if c in set(X["Kitchen Qual"].dropna())]
    e_choices  = [c for c in ["Ex","Gd","TA","Fa"]      if c in set(X["Exter Qual"].dropna())]

    with st.expander("🏡 House characteristics", expanded=True):
        col_l, col_r = st.columns(2)

        with col_l:
            v = {}
            v["Neighborhood"]   = st.selectbox(LABELS["Neighborhood"], nb_choices,
                                               index=nb_choices.index(D["Neighborhood"]))
            v["Overall Qual"]   = st.slider(LABELS["Overall Qual"],   1, 10, D["Overall Qual"])
            v["Overall Cond"]   = st.slider(LABELS["Overall Cond"],   1,  9, D["Overall Cond"])
            v["Year Built"]     = st.slider(LABELS["Year Built"],  1880, 2010, D["Year Built"])
            v["Year Remod/Add"] = st.slider(LABELS["Year Remod/Add"], 1950, 2010, D["Year Remod/Add"])
            v["1st Flr SF"]     = st.slider(LABELS["1st Flr SF"],   300, 2500, D["1st Flr SF"],  step=10)
            v["2nd Flr SF"]     = st.slider(LABELS["2nd Flr SF"],     0, 1500, D["2nd Flr SF"],  step=10)
            v["Total Bsmt SF"]  = st.slider(LABELS["Total Bsmt SF"],  0, 2500, D["Total Bsmt SF"], step=10)
            v["Lot Area"]       = st.slider(LABELS["Lot Area"],    1500,30000, D["Lot Area"],     step=100)

        with col_r:
            v["Bedroom AbvGr"]  = st.slider(LABELS["Bedroom AbvGr"],  0, 6, D["Bedroom AbvGr"])
            v["Full Bath"]      = st.slider(LABELS["Full Bath"],      0, 4, D["Full Bath"])
            v["Half Bath"]      = st.slider(LABELS["Half Bath"],      0, 2, D["Half Bath"])
            v["Garage Cars"]    = st.slider(LABELS["Garage Cars"],    0, 4, D["Garage Cars"])
            v["Fireplaces"]     = st.slider(LABELS["Fireplaces"],     0, 3, D["Fireplaces"])
            v["Kitchen Qual"]   = st.selectbox(LABELS["Kitchen Qual"], q_choices,
                                              index=q_choices.index(D["Kitchen Qual"])
                                              if D["Kitchen Qual"] in q_choices else 0)
            v["Exter Qual"]     = st.selectbox(LABELS["Exter Qual"], e_choices,
                                              index=e_choices.index(D["Exter Qual"])
                                              if D["Exter Qual"] in e_choices else 0)
            v["Central Air"]    = st.selectbox(LABELS["Central Air"], ["Y","N"],
                                              index=0 if D["Central Air"] == "Y" else 1)

    # ── live prediction
    preds      = predict_all(v)
    final_pred = preds[final_name]
    lo         = final_pred * np.exp(q10)
    hi         = final_pred * np.exp(q90)

    # hero price card
    st.markdown(
        f'<div class="price-hero">'
        f'  <p style="opacity:0.7;font-size:0.9rem;margin-bottom:4px">Predicted sale price</p>'
        f'  <h1>${final_pred:,.0f}</h1>'
        f'  <p>80% range &nbsp;·&nbsp; ${lo:,.0f} – ${hi:,.0f}</p>'
        f'  <p style="font-size:0.78rem;opacity:0.65">Model: {final_name}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1.1, 1])

    with col_a:
        # ── sensitivity chart
        st.subheader("Why this price?")
        st.caption("One input changed at a time vs a typical Ames house. Not SHAP.")
        deltas = compute_deltas(v, final_pred)
        s = deltas[::-1]

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        colors = ["#2E7D32" if x >= 0 else "#B4530A" for x in s.values]
        bars = ax.barh(s.index, s.values, color=colors, height=0.6)
        ax.axvline(0, color="black", lw=0.8)
        for bar, val in zip(bars, s.values):
            ax.text(
                val + (max(s.values) * 0.01 if val >= 0 else min(s.values) * 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"${val:+,.0f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=7.5, color="#333",
            )
        ax.set_xlabel("Effect on price vs. typical house ($)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        # ── model comparison table
        st.subheader("All models side by side")
        cmp_df = pd.DataFrame({
            "Model":           list(preds.keys()),
            "Predicted ($)":   [f"${p:,.0f}" for p in preds.values()],
            "vs. final model": [f"{p - final_pred:+,.0f}" for p in preds.values()],
        })
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        st.info(
            "💡 **Jury 2 finding:** Ridge (regularised linear) beats Gradient Boosting "
            "once cleaning and feature engineering are done properly. In Jury 1, boosting "
            "only won because the linear pipeline was weak.",
            icon=None,
        )

    # save inputs to session state for the AI tab
    st.session_state["current_v"]          = v
    st.session_state["current_preds"]      = preds
    st.session_state["current_final_pred"] = final_pred
    st.session_state["current_lo"]         = lo
    st.session_state["current_hi"]         = hi
    st.session_state["current_deltas"]     = deltas

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – AI EXPLANATION (Gemini)
# ════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.markdown(
        "### 🤖 AI Explanation  "
        '<span class="ai-badge">JURY 3 PREVIEW</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Gemini reads the model's prediction and sensitivity data and explains it in plain English. "
        "You can ask follow-up questions. The regression model stays the predictor; GenAI only explains."
    )

    if not _gemini_available():
        st.warning(
            "**Gemini API key not configured.** "
            "Add `GEMINI_API_KEY` to your Streamlit secrets to enable this tab.\n\n"
            "👉 Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)",
            icon="🔑",
        )
        st.markdown("""
**To add your key on Streamlit Cloud:**
1. Go to your app dashboard → **⋮ menu → Settings → Secrets**
2. Add this (replace with your actual key):
```toml
GEMINI_API_KEY = "AIza..."
```
3. Save — the app will restart automatically.
""")
    else:
        # pull latest prediction from session state (set in tab 1)
        v          = st.session_state.get("current_v", D)
        preds      = st.session_state.get("current_preds", {})
        final_pred = st.session_state.get("current_final_pred", 0.0)
        lo         = st.session_state.get("current_lo", 0.0)
        hi         = st.session_state.get("current_hi", 0.0)
        deltas     = st.session_state.get("current_deltas", pd.Series(dtype=float))

        if not preds:
            st.info("Go to the **🔮 Predict** tab first and set your house details, then come back here.")
        else:
            context = build_context(v, preds, final_pred, lo, hi, deltas)

            # auto-explain button / new context detection
            context_changed = (context != st.session_state.last_context)
            col_btn1, col_btn2 = st.columns([1, 5])
            with col_btn1:
                explain_clicked = st.button("✨ Explain this prediction", type="primary")
            with col_btn2:
                if st.button("🗑️ Clear chat"):
                    st.session_state.chat_history = []
                    st.session_state.display_msgs = []
                    st.session_state.last_context = ""
                    st.rerun()

            if context_changed and st.session_state.display_msgs:
                st.info("ℹ️ House details changed. Click **Explain this prediction** for a fresh explanation.")

            if explain_clicked:
                st.session_state.chat_history = []
                st.session_state.display_msgs = []
                st.session_state.last_context = context

                first_prompt = (
                    f"Here is the house and prediction data:\n\n{context}\n\n"
                    "Please explain this prediction in plain English. Be specific about "
                    "the dollar amounts and what drives the price for this particular house."
                )

                with st.spinner("Gemini is thinking…"):
                    try:
                        reply = ask_gemini(SYSTEM_PROMPT, [], first_prompt)
                        st.session_state.chat_history = [
                            {"role": "user",  "parts": [first_prompt]},
                            {"role": "model", "parts": [reply]},
                        ]
                        st.session_state.display_msgs = [
                            {"role": "assistant", "content": reply},
                        ]
                    except Exception as e:
                        st.error(f"Gemini error: {e}")

            # render chat history
            for msg in st.session_state.display_msgs:
                with st.chat_message(msg["role"],
                                     avatar="🤖" if msg["role"] == "assistant" else "🧑"):
                    st.markdown(msg["content"])

            # follow-up chat input
            if st.session_state.display_msgs:
                follow_up = st.chat_input("Ask a follow-up question…")
                if follow_up:
                    st.session_state.display_msgs.append(
                        {"role": "user", "content": follow_up}
                    )
                    with st.chat_message("user", avatar="🧑"):
                        st.markdown(follow_up)

                    # inject current context as a reminder for follow-ups
                    augmented = (
                        f"[Current house context for reference]\n{context}\n\n"
                        f"User question: {follow_up}"
                    )
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.spinner("Thinking…"):
                            try:
                                reply = ask_gemini(
                                    SYSTEM_PROMPT,
                                    st.session_state.chat_history,
                                    augmented,
                                )
                                st.markdown(reply)
                                st.session_state.chat_history += [
                                    {"role": "user",  "parts": [augmented]},
                                    {"role": "model", "parts": [reply]},
                                ]
                                st.session_state.display_msgs.append(
                                    {"role": "assistant", "content": reply}
                                )
                            except Exception as e:
                                st.error(f"Gemini error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown(
        "Evaluated on the same **586 held-out test houses** (80/20 split, seed 42). "
        "One test house is a documented anomaly — a 4,676 sq ft *Partial* sale at "
        "\\$184,750 — that dominates RMSE for linear models because they extrapolate "
        "to ~\\$884K on it. Tree models cap their predictions. Both views are shown."
    )
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.caption(
        "**Key finding:** Jury 2 Ridge RMSE (\\$19,000) beats Jury 1 Gradient Boosting "
        "(\\$22,181) on the same test set. Better cleaning + regularisation, not a "
        "more complex model, drove the improvement."
    )

    st.divider()
    st.subheader(f"Permutation importance — {final_name}")
    st.caption(
        "How much does test RMSE increase when each input is randomly shuffled? "
        "Works for any model type, unlike tree-based feature importance."
    )
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    s2 = imp[::-1]
    ax2.barh(s2.index, s2.values, color="#1E2761", height=0.6)
    ax2.set_xlabel("Increase in RMSE ($) when feature is shuffled", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_title(f"Permutation importance — {final_name}", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – ABOUT & ROADMAP
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### How it works
`User inputs → typical-house defaults for ~60 other columns → feature engineering (leak-safe) → Ridge model → price`

The same `model.py` file powers both this app and the Colab notebook, so every
number shown here is real and reproducible.

---

### What changed since Jury 1

| Jury 1 weakness | Jury 2 fix |
|---|---|
| Blank values treated as lost data | Domain-aware cleaning: blank = feature absent |
| Raw columns only | 23 engineered features (total area, bathrooms, house age, quality × size) |
| Errors grew with price | Log-transformed target |
| One "Partial sale" anomaly distorted training | Removed from training only; results shown with and without it |
| One split, no tuning | 5-fold CV + RandomizedSearchCV |
| Unstable coefficients on rare categories | Rare-category grouping, Ridge and Lasso |
| Only tree-based importance | Permutation importance — works for any model |

---

### Honest limitations
- Trained on Ames, Iowa (2006–2010) sales only — not valid for other cities or years
- One Partial-sale anomaly dominates full-test RMSE for linear models; both views are reported
- The "Why this price?" chart is a simple one-at-a-time sensitivity check, **not SHAP**
- Absolute dollar errors still grow with price; percentage error is flat
- Top-quality tier (9–10) and Crawford neighbourhood have the largest errors

---

### Roadmap — Jury 3
- ✅ **GenAI explanation** (Gemini) — live in this app as a preview
- ⬜ SHAP values for rigorous, game-theoretically grounded explanations
- ⬜ Model stacking (blend Ridge + GB for lower variance)
- ⬜ Conversational follow-up with memory across sessions
""")
