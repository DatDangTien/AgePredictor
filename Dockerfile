# GPU image: onnxruntime-gpu 1.23 needs CUDA 12 + cuDNN 9.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
# python3.10 (ubuntu 22.04 default) + opencv-headless runtime libs (libgomp, glib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Models are mounted as a read-only volume at runtime (see docker-compose.yml).
ENV MODELS_DIR=/models
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
