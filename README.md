# 💹 AI Financial Advisor

**Author:** Anastasiia Shchegelskaia  
**Mentor:** Dr. Qingyang Xiao  
**License:** MIT

A Streamlit-ready AI financial decision-support application converted from the original Jupyter/Colab prototype. The app uses simulated personal-finance records to demonstrate budgeting analytics, machine-learning forecasts, neural-network investment projections, overspending-risk classification, statistical validation, interactive visualizations, and an advisor-agent recommendation layer.

> **Important:** This project is an educational AI/ML demonstration using synthetic data. It is not individualized investment, tax, legal, or financial advice.

## Core capabilities

- **Budget & spending analytics** across restaurant, transit, gas, tuition, shopping, housing, utilities, entertainment, health, and debt payments.
- **Next-month spend forecasting** with a Random Forest regressor.
- **12-month portfolio projection** with an `MLPRegressor` neural network.
- **Overspending-risk prediction** with a Random Forest classifier and probability output.
- **Statistical validation** with a chi-square test between simulated risk profile and overspending behavior.
- **Feature importance** explaining the strongest drivers of the budget forecast model.
- **Peer benchmarking** against simulated users in a similar income band.
- **AI advisor recommendations** that turn analytics into prioritized actions.
- **Interactive visualizations** including bar, histogram, forecast-line, feature-importance, and animated trend views.
- **Synthetic-data preview and CSV download** from the Streamlit application.

## Repository structure

```text
.
├── streamlit_app.py            # Streamlit Community Cloud entry point
├── finance_core.py             # Data simulation, ML models, statistics, charts, advisor logic
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md                   # Project and deployment documentation
├── .gitignore
├── .streamlit/
│   └── config.toml             # Streamlit theme/configuration
├── assets/
│   └── spending_trend.gif      # Generated trend animation
└── notebooks/
    └── 071226_Anastasiia_Project_AI_Financial_Advisor.ipynb
```

## Run locally

Python 3.12 is recommended to mirror the current default for new Streamlit Community Cloud deployments.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Run the app from the repository root:

```bash
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload all files and folders from this project ZIP to the repository root.
3. Sign in to Streamlit Community Cloud.
4. Create a new app and select your GitHub repository and branch.
5. Set the entry-point file to:

   ```text
   streamlit_app.py
   ```

6. In **Advanced settings**, select Python **3.12**.
7. Deploy the app. No API keys or secrets are required for this synthetic-data demo.

## AI/ML workflow

```text
Synthetic financial records
        ↓
Feature engineering
        ↓
┌──────────────────────────────────────┐
│ Random Forest Regression             │ → next-month spend
│ MLP Neural Network Regression        │ → 12-month portfolio value
│ Random Forest Classification         │ → overspending probability
└──────────────────────────────────────┘
        ↓
Model metrics + chi-square statistics
        ↓
Visual analytics + peer benchmark
        ↓
AI advisor recommendation layer
        ↓
Streamlit dashboard
```

## Data privacy

The current prototype trains and scores against a reproducible **synthetic** dataset generated in code. It does not connect to bank accounts, brokerage accounts, credit cards, or external financial APIs.

## Credits

- **Author:** Anastasiia Shchegelskaia
- **Mentor:** Dr. Qingyang Xiao

## License

This repository is licensed under the **MIT License**. See [LICENSE](LICENSE).
