# TTS box image: the web UI and /overlay on port 8765, voices and downloaded models in mounted folders.
FROM python:3.12-slim

# espeak-ng: Kokoro's fallback for English words it doesn't know; portaudio: sounddevice (--output local);
# build-essential: pyworld (RVC dependency) has no prebuilt wheel for Python 3.12 and compiles on install
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng libportaudio2 libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.1 torchaudio==2.11.0 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu126
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY static ./static

# Hugging Face (HuBERT, Kokoro) and torch caches go to a volume, so models download once
ENV HF_HOME=/cache/huggingface TORCH_HOME=/cache/torch PYTHONUNBUFFERED=1
EXPOSE 8765
ENTRYPOINT ["python", "app.py", "--host", "0.0.0.0", "--output", "web"]
