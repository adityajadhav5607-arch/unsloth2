FROM unsloth/unsloth:latest
WORKDIR /app
COPY train.py /app/train.py
RUN mkdir -p /outputs
CMD bash -lc 'set -ex; nvidia-smi || true; unsloth --version || true; python /app/train.py'
