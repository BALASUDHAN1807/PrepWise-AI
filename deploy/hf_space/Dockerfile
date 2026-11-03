# Hugging Face Docker Space image (UNTESTED here - docker was not available when this was written; see docs/huggingface_spaces.md)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends g++ default-jdk-headless curl ffmpeg util-linux \
    && rm -rf /var/lib/apt/lists/*
# Spaces run containers as a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_sm
# Bake the embedding model into the image so the first request does not download it.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

COPY --chown=user . .
# Free Spaces have ephemeral disk: data below is lost on restart unless you attach persistent storage (mounted at /data).
ENV DATABASE_URL=sqlite:////home/user/app/prepwise.db ENVIRONMENT=prod WHISPER_MODEL=tiny
EXPOSE 7860
CMD ["bash", "deploy/hf_space/start.sh"]
