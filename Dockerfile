# CPU image for the footfall pipeline. For an NVIDIA box, start from an
# nvidia/cuda base, install the CUDA torch wheel, and run with `--gpus all`.
FROM python:3.12-slim

# opencv-python (the full, non-headless build the repo pins) needs these
# shared libraries even when it never opens a window.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY footfall/ ./footfall/
COPY tools/ ./tools/
COPY config/ ./config/

# YOLO weights download here on first run; the event store and recordings
# land in output/. Mount both as volumes to persist them (see
# docker-compose.yml).
RUN mkdir -p models output

ENV PYTHONUNBUFFERED=1 \
    FOOTFALL_LOG_FORMAT=json \
    FOOTFALL_LOG_LEVEL=INFO

# Provide config/site.yaml (and config/secrets.yaml, or the secret env
# vars) at run time -- they are not baked into the image.
ENTRYPOINT ["python", "tools/run_site.py"]
CMD ["config/site.yaml"]
