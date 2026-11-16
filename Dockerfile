# The official serving image (ADR 0002, D-05). Build it for linux/amd64, so the model is trained on the official platform:
#   docker build --platform linux/amd64 -t engine-rul:0.1 .
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime, which the slim image leaves out
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY config ./config
COPY data ./data
RUN pip install --no-cache-dir --no-deps .

# Train while the image is built: the image carries exactly one model, trained on this platform
RUN rul-train

EXPOSE 8000
CMD ["uvicorn", "engine_rul.service:app", "--host", "0.0.0.0", "--port", "8000"]
