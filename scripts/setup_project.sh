#!/usr/bin/env bash
set -e

echo "📦 Setting up project..."

# 1. Install Git LFS if needed
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️ Git LFS not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y git-lfs
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git-lfs
    fi
    git lfs install
else
    git lfs install
fi

# 2. Pull LFS files
echo "⬇️  Downloading large assets via Git LFS..."
git lfs pull

# 3. Sync Python deps
echo "🐍 Syncing Python environment with uv..."
uv sync

echo "✅ Project setup complete!"
