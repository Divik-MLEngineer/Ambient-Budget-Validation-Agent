FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Install dependencies directly to be fast and robust
RUN uv pip install --system --no-cache-dir \
    fastapi \
    uvicorn \
    "google-adk[gcp]" \
    "google-cloud-aiplatform>=1.156.0" \
    httpx \
    google-auth

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
