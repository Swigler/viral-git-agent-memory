FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 memory \
    && mkdir /data \
    && chown memory:memory /data

WORKDIR /app
COPY --chown=memory:memory . /app

ENV MEMORY_DIR=/data HOST=0.0.0.0 PORT=3100
USER memory
VOLUME /data
EXPOSE 3100

CMD ["python", "api_server.py"]
