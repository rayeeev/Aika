FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    coreutils \
    ca-certificates \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 runner

WORKDIR /work
USER runner

ENTRYPOINT ["bash", "-lc"]
