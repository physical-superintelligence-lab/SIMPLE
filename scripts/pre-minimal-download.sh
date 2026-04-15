#!/usr/bin/env bash

set -e

# Configuration
HF_REPO="USC-PSI-Lab/SIMPLE"
HF_BRANCH="main"
HF_URL="https://huggingface.co/datasets/${HF_REPO}/resolve/${HF_BRANCH}"
DATA_DIR="data"
KEEP_ZIPS=true  # Default: keep zip files, use --cleanup to delete them

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup)
            KEEP_ZIPS=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cleanup      Delete zip files after extraction (default: keep them)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "📥 Downloading resource files from Hugging Face (${HF_REPO})..."

# Files to download
declare -a fixed_files=(
    "vMaterials_2.zip"
    "assets_graspnet.zip"
)

declare -a glob_patterns=(
    # "scenes_hssd_"
    "robots_"
)

# ============================================================================
# Hugging Face Download Functions
# ============================================================================

download_hf_file() {
    local filename=$1
    local url="${HF_URL}/${filename}"

    # Check if file already exists (resume case)
    if [ -f "$filename" ]; then
        echo "  📦 $filename (resuming...)"
    else
        echo "  📦 $filename"
    fi

    # Try wget first (has cleaner single progress bar), fall back to curl
    if command -v wget &> /dev/null; then
        # -c flag continues partial downloads
        if wget -q --show-progress -c -O "$filename" "$url" 2>&1; then
            echo "✅ Downloaded $filename"
            return 0
        fi
    else
        # Fallback to curl with resume support (-C -)
        if curl -L -f -C - -o "$filename" "$url" 2>&1; then
            echo "✅ Downloaded $filename"
            return 0
        fi
    fi

    echo "⚠️  Could not download ${filename}"
    return 1
}

download_hf_pattern() {
    local pattern=$1
    local prefix="${pattern%\*}"

    # Simple spinner
    local spinner='|/-\'
    local idx=0

    # Run Python in background
    python3 -c "
import sys
from huggingface_hub import list_repo_files
try:
    files = list_repo_files('${HF_REPO}', repo_type='dataset', revision='${HF_BRANCH}')
    for f in files:
        if f.startswith('${prefix}'):
            print(f)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
" > /tmp/hf_files_$$.txt 2>&1 &

    local pid=$!

    # Show spinner while waiting
    while kill -0 $pid 2>/dev/null; do
        printf "\r  ${spinner:$idx:1} Searching for files matching '$pattern'..."
        idx=$(( (idx + 1) % 4 ))
        sleep 0.2
    done

    wait $pid

    # Clear spinner line
    printf "\r  ✓ Searching for files matching '$pattern'...   \n"

    local files=$(cat /tmp/hf_files_$$.txt 2>/dev/null)
    rm -f /tmp/hf_files_$$.txt

    if [ -z "$files" ]; then
        echo "  ⚠️  No files found matching pattern: ${pattern}"
        return 1
    fi

    while IFS= read -r filename; do
        [ -z "$filename" ] && continue
        # Skip error messages
        [[ "$filename" == "Listing files..."* ]] && continue
        [[ "$filename" == "Found"* ]] && continue
        download_hf_file "$filename"
    done <<< "$files"
}

# ============================================================================
# Main Download Logic
# ============================================================================

echo ""
echo "Downloading big resource files..."
for file in "${fixed_files[@]}"; do
    download_hf_file "$file" || true
done

echo ""
echo "Downloading files matching glob patterns..."
for pattern in "${glob_patterns[@]}"; do
    download_hf_pattern "$pattern" || true
done

# ============================================================================
# Extract Files
# ============================================================================

echo ""
echo "📦 Extracting files..."

for zip_file in *.zip; do
    if [ -f "$zip_file" ]; then
        # Extract robots_*.zip, assets_graspnet.zip, and scenes_hssd_*.zip directly to data/
        if [[ "$zip_file" =~ ^robots_ ]] || [[ "$zip_file" == "assets_graspnet.zip" ]] || [[ "$zip_file" =~ ^scenes_hssd_ ]]; then
            echo "  Extracting $zip_file directly to data/..."
            unzip -q "$zip_file"
            if [ "$KEEP_ZIPS" = false ]; then
                rm "$zip_file"
            fi
            echo "✅ Extracted $zip_file"
        else
            dirname="${zip_file%.zip}"
            echo "  Extracting $zip_file into $dirname/..."
            mkdir -p "$dirname"
            unzip -q "$zip_file" -d "$dirname"
            if [ "$KEEP_ZIPS" = false ]; then
                rm "$zip_file"
            fi
            echo "✅ Extracted $dirname"
        fi
    fi
done

cd ..
echo ""
echo "✅ All resource files downloaded and extracted successfully."
echo "📂 Files are in: data/"
if [ "$KEEP_ZIPS" = true ]; then
    echo "💾 Zip files kept in data/ (use --cleanup to delete them)"
else
    echo "🗑️  Zip files deleted after extraction"
fi
