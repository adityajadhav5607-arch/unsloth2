FROM unsloth/unsloth:latest
USER root               # temporarily elevate for build
WORKDIR /app
COPY train.py /app/train.py
RUN mkdir -p /outputs && chown -R 1000:1000 /outputs
USER 1000               # drop back to default non-root user
CMD bash -lc 'set -ex; nvidia-smi || true; unsloth --version || true; python /app/train.py'
