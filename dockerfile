FROM python:3.10.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY streamlit_app.py .
COPY models/ models/

EXPOSE 8000
EXPOSE 8501

# Default to FastAPI, but can be overridden
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

