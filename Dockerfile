FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY dashboard.py run_local.sh ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "recommender_drift_monitor.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
