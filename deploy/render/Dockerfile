# Render free-tier image (512 MB RAM): single container, API on 127.0.0.1:8000, Streamlit UI on $PORT.
# Slim requirements: torch-based packages removed because they push memory past 512 MB (measured: API 837 MB with torch, 343 MB without).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ default-jdk-headless curl ffmpeg util-linux \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt && python -m spacy download en_core_web_sm
COPY . .
ENV PYTHONUNBUFFERED=1 ENVIRONMENT=prod DATABASE_URL=sqlite:////app/prepwise.db \
    EMBEDDING_BACKEND=hash VECTOR_STORE=memory ENABLE_TRANSFORMER_SENTIMENT=false WHISPER_MODEL=tiny
CMD ["bash", "deploy/hf_space/start.sh"]
