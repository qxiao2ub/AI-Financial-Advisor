"""Core analytics for the AI Financial Advisor.

Adapted from the original Colab/Gradio notebook for Streamlit Community Cloud.
Author: Anastasiia Shchegelskaia
Mentor: Dr. Qingyang Xiao
"""


import os
import io
import math
import textwrap
from pathlib import Path

import imageio.v2 as imageio

APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(APP_DIR / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ASSETS_DIR = APP_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

CATEGORY_COLUMNS = [
    "restaurant",
    "bus",
    "gas",
    "tuition",
    "shopping",
    "housing",
    "utilities",
    "entertainment",
    "health",
]

FEATURE_COLUMNS = [
    "income",
    "restaurant",
    "bus",
    "gas",
    "tuition",
    "shopping",
    "housing",
    "utilities",
    "entertainment",
    "health",
    "debt_payment",
    "investment_contribution",
    "current_portfolio",
    "market_return",
    "budget_target",
    "age",
    "household_size",
    "savings_goal",
    "risk_profile",
    "month_num",
    "spend_to_income",
    "category_volatility",
]

NUMERIC_FEATURES = [
    col for col in FEATURE_COLUMNS if col != "risk_profile"
]
CATEGORICAL_FEATURES = ["risk_profile"]

RISK_RETURN = {
    "Conservative": 0.045,
    "Balanced": 0.075,
    "Aggressive": 0.11,
}
RISK_VOLATILITY = {
    "Conservative": 0.02,
    "Balanced": 0.04,
    "Aggressive": 0.07,
}
RISK_OVERSPEND_BIAS = {
    "Conservative": -0.05,
    "Balanced": 0.0,
    "Aggressive": 0.05,
}


def simulate_financial_data(n_samples: int = 1800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    records = []

    for i in range(n_samples):
        dt = months[i % len(months)]
        month_num = int(dt.month)

        risk_profile = rng.choice(
            ["Conservative", "Balanced", "Aggressive"],
            p=[0.32, 0.45, 0.23],
        )
        age = int(np.clip(rng.normal(31, 8), 20, 65))
        household_size = int(rng.choice([1, 2, 3, 4], p=[0.45, 0.30, 0.17, 0.08]))

        base_income = rng.normal(5200, 1400)
        income = float(np.clip(
            base_income + household_size * 380 + (age - 28) * 28 + RISK_RETURN[risk_profile] * 1800,
            2600,
            13000,
        ))

        market_return = float(rng.normal(0.006 + RISK_RETURN[risk_profile] / 20, 0.018))
        student_factor = float(rng.binomial(1, 0.22))
        commuter_factor = float(rng.uniform(0.6, 1.5))
        urban_factor = float(rng.uniform(0.8, 1.3))
        seasonal_shock = 1.0 + 0.08 * math.sin((month_num - 1) / 12 * 2 * math.pi)

        housing = np.clip(income * rng.uniform(0.18, 0.28) + household_size * 120, 800, 3000)
        utilities = np.clip(housing * rng.uniform(0.10, 0.18), 120, 450)
        restaurant = np.clip(income * rng.uniform(0.03, 0.09) * seasonal_shock * (1 + RISK_OVERSPEND_BIAS[risk_profile]), 60, 900)
        bus = np.clip(55 * commuter_factor * rng.uniform(0.7, 1.7), 20, 220)
        gas = np.clip(110 * commuter_factor * urban_factor * rng.uniform(0.7, 2.0), 30, 420)
        tuition = np.clip(student_factor * rng.uniform(120, 1400), 0, 1400)
        shopping = np.clip(income * rng.uniform(0.03, 0.10) * seasonal_shock * (1 + RISK_OVERSPEND_BIAS[risk_profile]), 70, 1100)
        entertainment = np.clip(income * rng.uniform(0.02, 0.07) * seasonal_shock, 45, 700)
        health = np.clip(70 + household_size * 35 + rng.normal(45, 35), 40, 420)

        debt_payment = np.clip(income * rng.uniform(0.03, 0.13), 80, 1000)
        investment_contribution = np.clip(
            income * rng.uniform(0.05, 0.18) * (1 + RISK_RETURN[risk_profile]),
            100,
            2000,
        )
        current_portfolio = np.clip(
            rng.normal(12000 + age * 650 + household_size * 1800, 8500),
            1000,
            150000,
        )

        total_spend = float(
            restaurant + bus + gas + tuition + shopping + housing + utilities + entertainment + health + debt_payment
        )

        budget_target = float(
            np.clip(
                income * (
                    0.68
                    - 0.03 * (risk_profile == "Conservative")
                    + 0.02 * (risk_profile == "Aggressive")
                    + 0.015 * (household_size - 1)
                ),
                income * 0.52,
                income * 0.86,
            )
        )

        overspend_margin = (
            total_spend
            - budget_target
            + rng.normal(0, 140)
            + RISK_OVERSPEND_BIAS[risk_profile] * income
        )
        overspent = int(overspend_margin > 0)

        spend_to_income = float(total_spend / income)
        category_volatility = float(
            np.std([
                restaurant, bus, gas, tuition, shopping,
                housing, utilities, entertainment, health
            ]) / max(income, 1)
        )

        next_month_spend = float(
            np.clip(
                total_spend
                * (
                    1.01
                    + 0.03 * overspent
                    + 0.10 * market_return
                    + 0.02 * seasonal_shock
                    + 0.04 * (shopping / max(total_spend, 1))
                )
                + rng.normal(0, 180),
                800,
                12000,
            )
        )

        annual_return = RISK_RETURN[risk_profile] + rng.normal(0.01, RISK_VOLATILITY[risk_profile])
        future_investment_value = float(
            np.clip(
                current_portfolio * (1 + annual_return)
                + investment_contribution * 12 * (1 + annual_return / 2)
                - debt_payment * rng.uniform(0.05, 0.12),
                1000,
                300000,
            )
        )

        records.append({
            "date": dt,
            "month_num": month_num,
            "risk_profile": risk_profile,
            "age": age,
            "household_size": household_size,
            "income": round(income, 2),
            "market_return": round(market_return, 4),
            "restaurant": round(float(restaurant), 2),
            "bus": round(float(bus), 2),
            "gas": round(float(gas), 2),
            "tuition": round(float(tuition), 2),
            "shopping": round(float(shopping), 2),
            "housing": round(float(housing), 2),
            "utilities": round(float(utilities), 2),
            "entertainment": round(float(entertainment), 2),
            "health": round(float(health), 2),
            "debt_payment": round(float(debt_payment), 2),
            "investment_contribution": round(float(investment_contribution), 2),
            "current_portfolio": round(float(current_portfolio), 2),
            "budget_target": round(float(budget_target), 2),
            "total_spend": round(total_spend, 2),
            "next_month_spend": round(next_month_spend, 2),
            "future_investment_value": round(future_investment_value, 2),
            "overspent": overspent,
            "spend_to_income": round(spend_to_income, 4),
            "category_volatility": round(category_volatility, 4),
        })

    df = pd.DataFrame(records)
    df["monthly_savings"] = (df["income"] - df["total_spend"] - df["investment_contribution"]).round(2)
    df["savings_goal"] = (df["income"] * np.clip(0.10 + 0.08 * (df["risk_profile"] == "Conservative"), 0.10, 0.24)).round(2)
    df["dominant_category"] = df[CATEGORY_COLUMNS].idxmax(axis=1)
    return df


def train_models(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y_spend = df["next_month_spend"]
    y_invest = df["future_investment_value"]
    y_over = df["overspent"]

    X_train, X_test, y_spend_train, y_spend_test = train_test_split(
        X, y_spend, test_size=0.20, random_state=42
    )
    _, _, y_invest_train, y_invest_test = train_test_split(
        X, y_invest, test_size=0.20, random_state=42
    )
    _, _, y_over_train, y_over_test = train_test_split(
        X, y_over, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    spend_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    investment_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=220,
                    early_stopping=True,
                    learning_rate_init=0.002,
                    random_state=42,
                ),
            ),
        ]
    )

    overspend_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    spend_model.fit(X_train, y_spend_train)
    investment_model.fit(X_train, y_invest_train)
    overspend_model.fit(X_train, y_over_train)

    spend_preds = spend_model.predict(X_test)
    invest_preds = investment_model.predict(X_test)
    over_preds = overspend_model.predict(X_test)

    metrics_df = pd.DataFrame([
        {
            "Model": "Budget Forecast (Random Forest)",
            "Task": "Regression",
            "R2": round(r2_score(y_spend_test, spend_preds), 4),
            "MAE": round(mean_absolute_error(y_spend_test, spend_preds), 2),
            "RMSE": round(math.sqrt(mean_squared_error(y_spend_test, spend_preds)), 2),
        },
        {
            "Model": "Investment Forecast (Neural Net)",
            "Task": "Regression",
            "R2": round(r2_score(y_invest_test, invest_preds), 4),
            "MAE": round(mean_absolute_error(y_invest_test, invest_preds), 2),
            "RMSE": round(math.sqrt(mean_squared_error(y_invest_test, invest_preds)), 2),
        },
        {
            "Model": "Overspending Risk (Random Forest)",
            "Task": "Classification",
            "Accuracy": round(accuracy_score(y_over_test, over_preds), 4),
            "Precision": round(precision_score(y_over_test, over_preds), 4),
            "Recall": round(recall_score(y_over_test, over_preds), 4),
            "F1": round(f1_score(y_over_test, over_preds), 4),
        },
    ]).fillna("—")

    chi2_table = pd.crosstab(df["risk_profile"], df["overspent"])
    chi2_stat, chi2_p, chi2_dof, _ = chi2_contingency(chi2_table)
    chi2_summary = pd.DataFrame([
        {
            "Chi-square statistic": round(float(chi2_stat), 4),
            "p-value": round(float(chi2_p), 6),
            "Degrees of freedom": int(chi2_dof),
            "Test": "risk_profile vs overspent",
        }
    ])

    spend_feature_names = spend_model.named_steps["prep"].get_feature_names_out()
    spend_importance = spend_model.named_steps["model"].feature_importances_
    feature_importance = pd.DataFrame({
        "feature": spend_feature_names,
        "importance": spend_importance,
    }).sort_values("importance", ascending=False).head(12)

    return {
        "spend_model": spend_model,
        "investment_model": investment_model,
        "overspend_model": overspend_model,
        "metrics_df": metrics_df,
        "chi2_summary": chi2_summary,
        "feature_importance": feature_importance,
        "X_test": X_test,
    }


DATASET = simulate_financial_data()
TRAINED = train_models(DATASET)


def create_animation(df: pd.DataFrame, output_path: Path) -> str:
    if output_path.exists():
        return str(output_path)

    monthly = (
        df.groupby(["date"])[CATEGORY_COLUMNS]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    frame_index = list(range(1, len(monthly) + 1, 2))
    if frame_index[-1] != len(monthly):
        frame_index.append(len(monthly))

    frames = []
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for idx in frame_index:
        ax.clear()
        subset = monthly.iloc[:idx]
        for col in ["restaurant", "gas", "shopping", "housing", "entertainment"]:
            ax.plot(subset["date"], subset[col], marker="o", label=col.title())
        ax.set_title("Animated Spending Trend in Synthetic Data")
        ax.set_ylabel("Average monthly spend ($)")
        ax.set_xlabel("Month")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=95)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))
        buffer.close()

    imageio.mimsave(output_path, frames, duration=0.60)
    plt.close(fig)
    return str(output_path)


TREND_GIF_PATH = create_animation(DATASET, ASSETS_DIR / "spending_trend.gif")


def make_bar_chart(category_spend: dict) -> go.Figure:
    data = pd.DataFrame({
        "Category": [key.replace("_", " ").title() for key in category_spend.keys()],
        "Amount": list(category_spend.values())
    }).sort_values("Amount", ascending=False)
    fig = px.bar(
        data,
        x="Category",
        y="Amount",
        title="Current Spending by Category",
        text="Amount",
    )
    fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
    fig.update_layout(height=380, xaxis_title="", yaxis_title="USD")
    return fig


def make_histogram(current_spend: float) -> go.Figure:
    fig = px.histogram(
        DATASET,
        x="total_spend",
        nbins=35,
        title="Peer Distribution of Monthly Spend (Synthetic Population)",
        marginal="box",
    )
    fig.add_vline(x=current_spend, line_dash="dash", annotation_text="Your spend")
    fig.update_layout(height=360, xaxis_title="Monthly spend (USD)", yaxis_title="Count")
    return fig


def make_forecast_line(current_spend: float, next_spend: float, savings_goal: float, budget_target: float) -> go.Figure:
    months = ["Now", "M+1", "M+2", "M+3", "M+4", "M+5", "M+6"]
    projected = [current_spend]
    for i in range(1, len(months)):
        drift = 1 - min(0.025, 0.006 * i) if next_spend > budget_target else 1 + 0.004 * i
        projected.append(max(0, next_spend * drift))

    forecast_df = pd.DataFrame({
        "Month": months,
        "Projected Spend": projected,
        "Budget Target": [budget_target] * len(months),
        "Savings Goal": [savings_goal] * len(months),
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["Month"], y=forecast_df["Projected Spend"], mode="lines+markers", name="Projected Spend"))
    fig.add_trace(go.Scatter(x=forecast_df["Month"], y=forecast_df["Budget Target"], mode="lines", name="Budget Target"))
    fig.add_trace(go.Scatter(x=forecast_df["Month"], y=forecast_df["Savings Goal"], mode="lines", name="Savings Goal"))
    fig.update_layout(title="6-Month Budget Trajectory", height=380, yaxis_title="USD")
    return fig


def make_feature_chart() -> go.Figure:
    fi = TRAINED["feature_importance"].copy()
    fi["feature"] = fi["feature"].str.replace("_", " ").str.title()
    fig = px.bar(
        fi.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Top Features Driving Budget Forecast Model",
    )
    fig.update_layout(height=420, xaxis_title="Importance score", yaxis_title="")
    return fig


def build_peer_summary(income: float, total_spend: float) -> str:
    peer_income_band = DATASET[np.abs(DATASET["income"] - income) <= 800]
    if peer_income_band.empty:
        peer_income_band = DATASET.copy()

    peer_mean = float(peer_income_band["total_spend"].mean())
    delta = total_spend - peer_mean
    direction = "above" if delta > 0 else "below"

    return (
        f"Peer benchmark: users near your income level spend about "
        f"${peer_mean:,.0f}/month on average, and your current spending is "
        f"${abs(delta):,.0f} {direction} that benchmark."
    )


def advisor_agent(
    income: float,
    total_spend: float,
    next_spend: float,
    overspend_prob: float,
    invest_12m: float,
    investment_contribution: float,
    budget_target: float,
    current_portfolio: float,
    risk_profile: str,
    category_spend: dict,
) -> str:
    top_three = sorted(category_spend.items(), key=lambda x: x[1], reverse=True)[:3]
    savings_room = income - total_spend - investment_contribution
    suggestions = []

    if total_spend > budget_target:
        suggestions.append(
            f"Budget control alert: your spending is ${total_spend - budget_target:,.0f} above the modeled budget target. "
            f"Cap the highest flexible category first: {top_three[0][0].title()}."
        )
    else:
        suggestions.append(
            "Budget control status: you are currently within the modeled budget range. Keep the same discipline next month."
        )

    if overspend_prob >= 0.60:
        suggestions.append(
            f"Risk signal: the overspending classifier estimates a {overspend_prob:.0%} chance of exceeding budget next month. "
            "A 10% cut to shopping and restaurant spend would materially reduce that probability."
        )
    else:
        suggestions.append(
            f"Risk signal: the classifier gives only a {overspend_prob:.0%} overspending risk next month. "
            "This leaves room to increase savings or invest more aggressively."
        )

    projected_gain = invest_12m - current_portfolio - investment_contribution * 12
    if projected_gain > 0:
        suggestions.append(
            f"Investment outlook: with a {risk_profile.lower()} profile, the 12-month portfolio forecast is "
            f"${invest_12m:,.0f}, implying roughly ${projected_gain:,.0f} of modeled growth after contributions."
        )
    else:
        suggestions.append(
            "Investment outlook: the model expects limited growth at your current contribution level. Consider increasing recurring contributions or lowering debt drag."
        )

    if savings_room < income * 0.08:
        suggestions.append(
            "Liquidity note: projected free cash after spending and investing is tight. Build a larger emergency buffer before increasing discretionary spending."
        )
    else:
        suggestions.append(
            "Liquidity note: your post-spending cash buffer looks healthy enough to support both savings and investment contributions."
        )

    suggestions.append(
        "Action plan: prioritize trimming "
        + ", ".join([name.title() for name, _ in top_three[:2]])
        + " if you need the fastest budget improvement."
    )

    return "\n".join([f"{i+1}. {item}" for i, item in enumerate(suggestions)])


def score_user(
    income,
    restaurant,
    bus,
    gas,
    tuition,
    shopping,
    housing,
    utilities,
    entertainment,
    health,
    debt_payment,
    investment_contribution,
    current_portfolio,
    budget_target,
    savings_goal,
    risk_profile,
    age,
    household_size,
    month_num,
):
    category_spend = {
        "restaurant": restaurant,
        "bus": bus,
        "gas": gas,
        "tuition": tuition,
        "shopping": shopping,
        "housing": housing,
        "utilities": utilities,
        "entertainment": entertainment,
        "health": health,
    }

    total_spend = float(sum(category_spend.values()) + debt_payment)
    spend_to_income = float(total_spend / max(income, 1))
    category_volatility = float(np.std(list(category_spend.values())) / max(income, 1))

    features = pd.DataFrame([{
        "income": income,
        "restaurant": restaurant,
        "bus": bus,
        "gas": gas,
        "tuition": tuition,
        "shopping": shopping,
        "housing": housing,
        "utilities": utilities,
        "entertainment": entertainment,
        "health": health,
        "debt_payment": debt_payment,
        "investment_contribution": investment_contribution,
        "current_portfolio": current_portfolio,
        "market_return": float(DATASET["market_return"].mean()),
        "budget_target": budget_target,
        "age": age,
        "household_size": household_size,
        "savings_goal": savings_goal,
        "risk_profile": risk_profile,
        "month_num": int(month_num),
        "spend_to_income": spend_to_income,
        "category_volatility": category_volatility,
    }])

    next_spend = float(TRAINED["spend_model"].predict(features)[0])
    invest_12m = float(TRAINED["investment_model"].predict(features)[0])
    overspend_proba = float(TRAINED["overspend_model"].predict_proba(features)[0][1])

    free_cash = income - total_spend - investment_contribution
    savings_rate = free_cash / max(income, 1)

    summary = pd.DataFrame([
        ["Current total spend", f"${total_spend:,.2f}"],
        ["Current investment contribution", f"${investment_contribution:,.2f}"],
        ["Predicted next-month spend", f"${next_spend:,.2f}"],
        ["Predicted 12-month portfolio value", f"${invest_12m:,.2f}"],
        ["Overspending risk", f"{overspend_proba:.1%}"],
        ["Free cash after spend + investing", f"${free_cash:,.2f}"],
        ["Savings rate", f"{savings_rate:.1%}"],
        ["Budget target gap", f"${(budget_target - total_spend):,.2f}"],
    ], columns=["Metric", "Value"])

    recommendations = advisor_agent(
        income=income,
        total_spend=total_spend,
        next_spend=next_spend,
        overspend_prob=overspend_proba,
        invest_12m=invest_12m,
        investment_contribution=investment_contribution,
        budget_target=budget_target,
        current_portfolio=current_portfolio,
        risk_profile=risk_profile,
        category_spend=category_spend,
    )

    executive_text = textwrap.dedent(f"""
    ### Financial AI Advisor Summary
    - **Monthly income:** ${income:,.0f}
    - **Current total spending:** ${total_spend:,.0f}
    - **Modeled budget target:** ${budget_target:,.0f}
    - **Predicted next-month spend:** ${next_spend:,.0f}
    - **12-month investment projection:** ${invest_12m:,.0f}
    - **Overspending probability:** {overspend_proba:.1%}

    {build_peer_summary(income, total_spend)}
    """)

    return (
        executive_text,
        summary,
        TRAINED["metrics_df"],
        TRAINED["chi2_summary"],
        make_bar_chart(category_spend),
        make_histogram(total_spend),
        make_forecast_line(total_spend, next_spend, savings_goal, budget_target),
        recommendations,
        TREND_GIF_PATH,
        make_feature_chart(),
        DATASET.head(15),
    )

