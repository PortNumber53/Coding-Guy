FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    nodejs npm \
    golang-go \
    git curl wget grep findutils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
