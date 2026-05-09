FROM ubuntu:22.04

# Install Node.js 22.x LTS from NodeSource (Ubuntu 22.04 ships v12, Vite 8 requires Node 20+, Wrangler requires Node 22+)
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg ca-certificates \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
 python3 python3-pip python3-venv \
 golang-go \
 git curl wget grep findutils \
 build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
