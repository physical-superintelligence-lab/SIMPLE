#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

export UV_HTTP_TIMEOUT=300

echo "📦 Initializing IsaacLab submodule..."
git submodule update --init --recursive third_party/IsaacLab

source .venv/bin/activate

echo "💾 Pulling Git LFS objects for IsaacLab (if any)..."
cd third_party/IsaacLab
if git lfs > /dev/null 2>&1; then
    git lfs pull || echo "⚠️ Git LFS not configured or no LFS files found."
else
    echo "⚠️ Git LFS not installed. Skipping LFS pull."
fi
cd ../..

echo "🧰 Installing system dependencies..."
# sudo apt update
# sudo apt install -y cmake build-essential

# # If isaacsim is installed in uv, link it
# if uv pip show isaacsim-rl > /dev/null 2>&1; then
#     echo "🔗 Linking Isaac Sim from uv environment..."
#     cd third_party/IsaacLab
#     ./isaaclab.sh --link  $(uv run which python)
#     cd ../..
# fi

echo "⚙️ Running IsaacLab installation script..."
cd third_party/IsaacLab
./isaaclab.sh --install none
cd ../..

echo "✅ IsaacLab installed successfully."