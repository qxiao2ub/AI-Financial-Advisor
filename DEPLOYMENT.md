# Streamlit Community Cloud Deployment Checklist

## GitHub
- Upload the entire repository contents without changing the folder structure.
- Confirm `streamlit_app.py` and `requirements.txt` are at the repository root.
- Keep `.streamlit/config.toml` in the `.streamlit` directory.

## Streamlit Community Cloud
- Repository: choose the GitHub repository containing this project.
- Branch: typically `main`.
- Main file path: `streamlit_app.py`.
- Python: 3.12 recommended.
- Secrets: none required for the current synthetic-data version.

## Expected first launch
The application initializes a synthetic dataset and trains three models on startup:
1. Random Forest budget regressor
2. MLP neural-network investment regressor
3. Random Forest overspending classifier

After initialization, the dashboard exposes the Live Advisor, Model Diagnostics, Synthetic Data, and About tabs.
