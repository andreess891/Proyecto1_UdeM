Ejecución

1. Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
2. .\.venv\Scripts\Activate.ps1
3. uv run python -m streamlit run app.py --server.headless true
4. http://localhost:8501
