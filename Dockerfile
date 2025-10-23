FROM unsloth/unsloth:latest

# Use root just for build steps
USER root

WORKDIR /app
COPY train.py /app/train.py

# create an /outputs folder and hand ownership back to the default user (1000)
RUN mkdir -p /outputs && chown -R 1000:1000 /outputs

# Drop back to the normal user
USER 1000

CMD bash -lc 'set -ex; nvidia-smi || true; unsloth --version || true; python /app/train.py'
