FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MM_PIPELINE_CONFIG=/app/config/ebola.yml

WORKDIR /app

# Build with mm_pipeline_2 itself as the Docker context:
# docker build -t mm-pipeline-2 .
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY config ./config

ENTRYPOINT ["mm-pipeline"]
CMD ["--help"]
