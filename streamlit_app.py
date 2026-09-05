from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Financial Advisor | Anastasiia Shchegelskaia",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Importing the core module initializes the reproducible synthetic dataset and
# trains the notebook's ML models once per Python process.
with st.spinner("Initializing the AI financial analytics engine..."):
    from finance_core import (
        DATASET,
        TRAINED,
        TREND_GIF_PATH,
        make_feature_chart,
        score_user,
    )

APP_DIR = Path(__file__).resolve().parent

CUSTOM_CSS = """
<style>
:root {
  --card-border: rgba(120, 120, 120, .16);
}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
.hero {
  padding: 1.55rem 1.7rem;
  border: 1px solid var(--card-border);
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(76,110,245,.14), rgba(32,201,151,.10));
  margin-bottom: 1rem;
}
.hero h1 {margin:0 0 .25rem 0; font-size:2.25rem;}
.hero p {margin:.15rem 0; font-size:1.02rem; opacity:.88;}
.credit {
  display:inline-block; margin-top:.65rem; padding:.35rem .65rem;
  border-radius:999px; border:1px solid var(--card-border); font-size:.88rem;
}
.section-note {
  border-left: 4px solid #4c6ef5;
  padding: .75rem 1rem;
  background: rgba(76,110,245,.06);
  border-radius: 8px;
}
.footer {
  margin-top:2.5rem; padding-top:1.2rem; border-top:1px solid var(--card-border);
  text-align:center; font-size:.9rem; opacity:.78;
}
[data-testid="stMetric"] {
  border:1px solid var(--card-border); border-radius:16px; padding:.75rem 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>💹 AI Financial Advisor</h1>
      <p>Machine-learning decision support for budgeting, spending behavior, and investment forecasting.</p>
      <span class="credit"><b>Author:</b> Anastasiia Shchegelskaia &nbsp; • &nbsp; <b>Mentor:</b> Dr. Qingyang Xiao</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Educational AI/ML demonstration based on simulated personal-finance data. "
    "It is not individualized investment, tax, legal, or financial advice."
)

with st.sidebar:
    st.header("Financial profile")
    income = st.slider("Monthly income ($)", 2500, 15000, 6500, 100)
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
    age = st.slider("Age", 20, 65, 30, 1)
    household_size = st.slider("Household size", 1, 4, 2, 1)
    month_num = st.slider("Current month number", 1, 12, 6, 1)
    budget_target = st.slider("Budget target ($)", 1500, 10000, 4200, 50)
    savings_goal = st.slider("Monthly savings goal ($)", 100, 3000, 800, 25)
    investment_contribution = st.slider("Monthly investment contribution ($)", 100, 2500, 600, 25)
    current_portfolio = st.slider("Current portfolio value ($)", 1000, 150000, 25000, 500)
    debt_payment = st.slider("Debt payment ($)", 0, 1500, 350, 25)

    st.divider()
    st.subheader("Spending categories")
    restaurant = st.slider("Restaurant", 0, 1200, 250, 10)
    bus = st.slider("Bus", 0, 300, 70, 5)
    gas = st.slider("Gas", 0, 500, 140, 5)
    tuition = st.slider("Tuition", 0, 1500, 150, 10)
    shopping = st.slider("Shopping", 0, 1200, 350, 10)
    housing = st.slider("Housing / Rent", 500, 3500, 1700, 25)
    utilities = st.slider("Utilities", 50, 600, 220, 5)
    entertainment = st.slider("Entertainment", 0, 900, 180, 5)
    health = st.slider("Health", 0, 500, 120, 5)

    run_btn = st.button("Run Financial Advisor", type="primary", use_container_width=True)
    st.caption("Inputs update the model analysis when you click the button.")

DEFAULT_INPUTS = dict(
    income=income,
    restaurant=restaurant,
    bus=bus,
    gas=gas,
    tuition=tuition,
    shopping=shopping,
    housing=housing,
    utilities=utilities,
    entertainment=entertainment,
    health=health,
    debt_payment=debt_payment,
    investment_contribution=investment_contribution,
    current_portfolio=current_portfolio,
    budget_target=budget_target,
    savings_goal=savings_goal,
    risk_profile=risk_profile,
    age=age,
    household_size=household_size,
    month_num=month_num,
)

if "advisor_inputs" not in st.session_state:
    st.session_state.advisor_inputs = DEFAULT_INPUTS.copy()
if run_btn:
    st.session_state.advisor_inputs = DEFAULT_INPUTS.copy()

result = score_user(**st.session_state.advisor_inputs)
(
    executive_text,
    summary_table,
    metrics_df,
    chi2_df,
    spend_bar,
    spend_hist,
    budget_line,
    recommendations,
    trend_gif_path,
    feature_plot,
    dataset_preview,
) = result

# Parse model outputs from the same summary table produced by the notebook.
summary_map = dict(zip(summary_table["Metric"], summary_table["Value"]))

live_tab, diagnostics_tab, data_tab, about_tab = st.tabs(
    ["📊 Live Advisor", "🧠 Model Diagnostics", "🗃️ Synthetic Data", "ℹ️ About"]
)

with live_tab:
    st.markdown(executive_text)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Next-month spend", summary_map.get("Predicted next-month spend", "—"))
    m2.metric("12-month portfolio", summary_map.get("Predicted 12-month portfolio value", "—"))
    m3.metric("Overspending risk", summary_map.get("Overspending risk", "—"))
    m4.metric("Free cash", summary_map.get("Free cash after spend + investing", "—"))

    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.subheader("AI advisor action plan")
        st.markdown(recommendations)
    with c2:
        st.subheader("Financial summary")
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

    st.subheader("Spending & budget analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(spend_bar, use_container_width=True)
    with c2:
        st.plotly_chart(spend_hist, use_container_width=True)
    st.plotly_chart(budget_line, use_container_width=True)

    st.subheader("Animated spending trend")
    trend_path = Path(trend_gif_path)
    if trend_path.exists():
        st.image(str(trend_path), caption="Average category spending over time in the synthetic training population")
    else:
        st.info("Trend animation will be generated when the analytics engine initializes.")

with diagnostics_tab:
    st.markdown(
        """
        <div class="section-note">
        The notebook uses supervised learning for next-month budget forecasting, a neural network for
        12-month investment-value forecasting, a Random Forest classifier for overspending risk, and a
        chi-square test for an interpretable relationship between risk profile and overspending behavior.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Model performance")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(feature_plot, use_container_width=True)
    with c2:
        st.subheader("Statistical validation")
        st.dataframe(chi2_df, use_container_width=True, hide_index=True)
        st.write(
            "The chi-square test in the source notebook evaluates whether simulated risk profile and "
            "overspending status are statistically associated."
        )

    st.subheader("AI pipeline")
    st.markdown(
        """
        1. **Synthetic finance generation** — creates reproducible records for income, spending, debt, savings, and market conditions.  
        2. **Feature engineering** — builds spend-to-income and category-volatility features.  
        3. **Budget forecasting** — Random Forest regression predicts next-month spending.  
        4. **Investment forecasting** — an MLP neural network projects a 12-month portfolio value.  
        5. **Risk classification** — Random Forest classification estimates overspending probability.  
        6. **Statistical validation** — chi-square analysis tests behavioral relationships.  
        7. **Advisor agent** — converts model outputs into prioritized, human-readable recommendations.
        """
    )

with data_tab:
    st.subheader("Synthetic finance dataset")
    st.write(
        f"The demo trains on **{len(DATASET):,} synthetic records**. No bank account, brokerage, or personal financial data is collected by this app."
    )
    preview_count = st.slider("Rows to preview", 10, 100, 25, 5, key="preview_rows")
    st.dataframe(DATASET.head(preview_count), use_container_width=True, hide_index=True)

    csv = DATASET.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download synthetic dataset (CSV)",
        data=csv,
        file_name="synthetic_financial_advisor_data.csv",
        mime="text/csv",
    )

with about_tab:
    st.subheader("Project credits")
    st.markdown(
        """
        **Author:** Anastasiia Shchegelskaia  
        **Mentor:** Dr. Qingyang Xiao

        This Streamlit application is derived from the project's original Jupyter/Colab notebook and
        preserves its main financial-analysis, machine-learning, neural-network, visualization, and
        recommendation workflow while adapting the interface for Streamlit Community Cloud.
        """
    )

    st.subheader("Technology stack")
    st.markdown(
        "**Python · Streamlit · pandas · NumPy · scikit-learn · SciPy · Plotly · Matplotlib · ImageIO**"
    )

    st.subheader("License")
    st.write("Released under the MIT License. See the repository's LICENSE file for the full text.")

st.markdown(
    """
    <div class="footer">
      AI Financial Advisor • Author: Anastasiia Shchegelskaia • Mentor: Dr. Qingyang Xiao • MIT License
    </div>
    """,
    unsafe_allow_html=True,
)
