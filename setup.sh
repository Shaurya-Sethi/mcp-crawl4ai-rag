#!/usr/bin/env bash
# Simple helper script to set up a local development environment
# Installs uv, creates a virtual environment, installs dependencies,
# and runs crawl4ai-setup.
set -euo pipefail

pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .
crawl4ai-setup

echo "Setup complete. Activate the environment with 'source .venv/bin/activate'"
