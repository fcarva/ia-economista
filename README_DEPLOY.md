# Deployment Guide for AI Economist

This project is configured for deployment on **Streamlit Cloud**, **Docker**, or **Railway/Heroku**.

## 1. Streamlit Cloud (Recommended)
The easiest way to host the dashboard.

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your account and select the repository.
4. Settings:
   - **Main file path**: `dashboard/Home.py`
   - **Python version**: `3.12`
5. Click **Deploy**.

## 2. Docker
To build and run the container locally:

```bash
# Build
docker build -t ai-economist .

# Run
docker run -p 8501:8501 ai-economist
```

### 3. Run Locally
```bash
# Linux/Mac
streamlit run dashboard/Home.py

# Windows (if streamlit command not found)
python -m streamlit run dashboard/Home.py
```
**Access**: [http://localhost:8501](http://localhost:8501).

## 4. Manual / Local
Ensure you have the dependencies installed:

```bash
pip install -r requirements.txt
python -m streamlit run dashboard/Home.py
```

## Structure
- `dashboard/`: Contains the UI code.
- `models/`: Trained GNN models.
- `data/`: Local cache (will be regenerated if missing, except for model weights which must be Git LFS or re-trained).
