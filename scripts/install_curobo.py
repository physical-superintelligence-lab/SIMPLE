"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

# scripts/setup_curobo.py

import subprocess
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUROBO_DIR = PROJECT_ROOT / "third_party" / "curobo"

def run(cmd, **kwargs):
    print(f"🔧 Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)

def main():
    # Step 1: Initialize submodule if needed
    if not CUROBO_DIR.exists() or not (CUROBO_DIR / "pyproject.toml").exists():
        print("📥 Initializing cuRobo submodule...")
        run(["git", "submodule", "update", "--init", "--recursive", "third_party/curobo"])
    else:
        print("✅ cuRobo submodule already initialized.")

    # Step 2: Install cuRobo into current uv environment
    print("📦 Installing cuRobo...")
    run([
        "uv", "pip", "install",
        "--no-build-isolation", "-e", str(CUROBO_DIR)
    ])

    print("🎉 cuRobo installed successfully.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}", file=sys.stderr)
        sys.exit(1)
