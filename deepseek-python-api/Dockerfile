FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    TOKENS_FILE=/data/deepseek_tokens.json

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir . \
    && mkdir -p /data \
    && chown 65534:65534 /data

USER 65534:65534
EXPOSE 8000
CMD ["deepseek-python-api"]
