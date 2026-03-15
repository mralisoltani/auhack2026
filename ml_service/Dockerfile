FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY ml_service/requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt
COPY src ./src
COPY ml_service ./ml_service
ENV PYTHONPATH=/app
WORKDIR /app/ml_service
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
