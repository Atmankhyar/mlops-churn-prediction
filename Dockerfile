FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

# MLflow model versions were registered with an absolute Windows host path as
# their artifact source (file:///C:/Users/.../mlflow-artifacts/...). Rather
# than rewrite that path in the registry database, recreate it as a symlink
# to the bind-mounted artifacts directory so mlflow.pyfunc.load_model() can
# resolve it unchanged inside the container.
RUN mkdir -p "/C:/Users/moham/Documents/MyProject/End-to-End MLOps" && \
    ln -s /mnt/mlflow-artifacts "/C:/Users/moham/Documents/MyProject/End-to-End MLOps/mlflow-artifacts"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]