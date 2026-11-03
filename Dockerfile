FROM python:3.11-slim

# g++ and a JDK let the sandbox run C++ and Java submissions; curl is for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends g++ default-jdk-headless curl ffmpeg util-linux \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_sm
COPY . .
ENV PYTHONUNBUFFERED=1 BACKEND_URL=http://backend:8000
EXPOSE 8000 8501
# Default: API. docker-compose overrides the command for the Streamlit service.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
