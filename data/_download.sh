#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
#
# Pre-download every asset that `simple.utils.resolve_data_path(..., auto_download=True)`
# would otherwise fetch lazily at runtime, and extract it into data/.
#
# The lazy path in src/simple/utils.py maps a relative data path to a zip name in the
# HF dataset repo USC-PSI-Lab/SIMPLE, downloads that zip into data/ and extracts it
# there. This script walks the same mapping ahead of time, for every asset the code
# can ask for:
#
#   group      zip(s)                          requested by
#   ---------- ------------------------------- --------------------------------------
#   robots     robots_<name>.zip               robots/*.py  (robot_cfg / mjcf / hand_yaml)
#   assets     assets_graspnet.zip             assets/graspnet.py, grasps/bodex.py
#              assets_objects.zip              assets/objects.py
#              assets_articulated.zip          assets/articulate.py
#   objaverse  assets_objaverse_NNNN.zip       assets/objaverse.py (1546 objects, sampled)
#   scenes     scenes_hssd_<name>.zip          scenes/hssd.py (names from hssd-scenes/config.yaml)
#   materials  vMaterials_2.zip                dr/material.py
#
# Everything is idempotent: an entry whose extracted directory is already present is
# skipped, so the script can be re-run to top up a partial data/ tree. An entry whose
# version in data/_versions.json is newer than its local stamp is fetched again.

set -uo pipefail

HF_REPO="USC-PSI-Lab/SIMPLE"
HF_REPO_TYPE="dataset"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"
SCENE_CONFIG="$PROJECT_DIR/src/simple/resources/hssd-scenes/config.yaml"
# Archives are staged here rather than straight into data/, so a root-owned
# data/.cache left behind by the docker flow cannot break the hub's lock files.
STAGE_DIR="$DATA_DIR/.predownload"

ALL_GROUPS=(robots assets objaverse scenes materials)
SEL_GROUPS=("${ALL_GROUPS[@]}")
KEEP_ZIPS=false
DRY_RUN=false
FORCE=false
LIST_ONLY=false
JOBS=8

usage() {
    cat <<EOF
Usage: data/_download.sh [OPTIONS]

Pre-download and extract all assets that SIMPLE would otherwise fetch on demand
from the Hugging Face dataset repo ${HF_REPO}.

Options:
  --only GROUPS    Comma-separated groups to fetch (default: all)
  --skip GROUPS    Comma-separated groups to leave out
  --list           Print what would be fetched (with download sizes) and exit
  --dry-run        Resolve the plan and print each action without downloading
  --force          Re-download and re-extract even if the target dir already exists
  --keep-zips      Keep the downloaded .zip files (default: delete after extraction)
  --jobs N         Parallel download workers (default: ${JOBS})
  --help           Show this message

Groups: ${ALL_GROUPS[*]}

Approximate download sizes: robots 0.4G, assets 0.5G, objaverse 2.2G,
scenes 20.5G, materials 3.2G  (~27G total; extracted size is similar).

Set HF_TOKEN (env or .env) if the repo ever requires authentication.

Examples:
  data/_download.sh                       # everything
  data/_download.sh --skip scenes         # skip the 20G HSSD scenes
  data/_download.sh --only robots,assets  # minimal tabletop set
  data/_download.sh --list
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --only)      IFS=',' read -r -a SEL_GROUPS <<< "$2"; shift 2 ;;
        --skip)
            IFS=',' read -r -a _skip <<< "$2"; shift 2
            _keep=()
            for g in "${SEL_GROUPS[@]}"; do
                _drop=false
                for s in "${_skip[@]}"; do [[ "$g" == "$s" ]] && _drop=true; done
                $_drop || _keep+=("$g")
            done
            SEL_GROUPS=("${_keep[@]}")
            ;;
        --list)      LIST_ONLY=true; shift ;;
        --dry-run)   DRY_RUN=true; shift ;;
        --force)     FORCE=true; shift ;;
        --keep-zips) KEEP_ZIPS=true; shift ;;
        --jobs)      JOBS="$2"; shift 2 ;;
        --help|-h)   usage; exit 0 ;;
        *) echo "Unknown option: $1"; echo "Use --help for usage information"; exit 1 ;;
    esac
done

for g in "${SEL_GROUPS[@]}"; do
    _known=false
    for k in "${ALL_GROUPS[@]}"; do [[ "$g" == "$k" ]] && _known=true; done
    if ! $_known; then
        echo "Unknown group: $g (known: ${ALL_GROUPS[*]})" >&2
        exit 1
    fi
done

# ============================================================================
# Python interpreter (needs huggingface_hub, already a project dependency)
# ============================================================================

PY=""
pick_python() {
    local candidates=()
    [[ -n "${PYTHON:-}" ]] && candidates+=("$PYTHON")
    candidates+=("$PROJECT_DIR/.venv/bin/python" "python3" "python")
    for c in "${candidates[@]}"; do
        command -v "$c" &> /dev/null || [[ -x "$c" ]] || continue
        if "$c" -c "import huggingface_hub" &> /dev/null; then PY="$c"; return 0; fi
    done
    if command -v uv &> /dev/null && (cd "$PROJECT_DIR" && uv run --quiet python -c "import huggingface_hub") &> /dev/null; then
        PY="uv-run"; return 0
    fi
    return 1
}

run_py() {  # run_py <<< script  (reads the script on stdin)
    if [[ "$PY" == "uv-run" ]]; then
        (cd "$PROJECT_DIR" && uv run --quiet python -)
    else
        "$PY" -
    fi
}

# The python snippets below are fed on stdin as heredocs, so the archive list has
# to travel through a file rather than a pipe.
LIST_FILE=""
cleanup_list_file() { [[ -n "$LIST_FILE" ]] && rm -f "$LIST_FILE"; }
trap cleanup_list_file EXIT

write_list_file() {
    LIST_FILE="$(mktemp "${TMPDIR:-/tmp}/simple-predownload.XXXXXX")"
    printf '%s\n' "${TODO_ZIPS[@]}" > "$LIST_FILE"
}

if ! pick_python; then
    echo "❌ No Python with huggingface_hub found." >&2
    echo "   Install the project deps (uv sync) or set PYTHON=/path/to/python." >&2
    exit 1
fi

command -v unzip &> /dev/null || { echo "❌ unzip is required but not installed." >&2; exit 1; }

if ! mkdir -p "$STAGE_DIR" 2>/dev/null || [[ ! -w "$STAGE_DIR" ]]; then
    echo "❌ Staging dir is not writable: $STAGE_DIR" >&2
    echo "   Fix ownership (e.g. sudo chown -R \"$(id -u):$(id -g)\" \"$DATA_DIR\") and retry." >&2
    exit 1
fi
if [[ ! -w "$DATA_DIR" ]]; then
    echo "❌ Data dir is not writable: $DATA_DIR" >&2
    exit 1
fi

# ============================================================================
# Manifest: "<zip name>|<directory that exists once extracted>"
#
# Every zip below is rooted at the data/ directory, exactly like the lazy
# `zip_ref.extractall(data_dir)` in resolve_data_path(), so extraction always
# targets data/ itself.
# ============================================================================

# robots/<name>/*.{yml,xml,usd} -> robots_<name>.zip
ROBOTS=(g1 g1_inspire g1_sonic vega_1 franka_fr3 aloha)

manifest_robots() {
    local r
    for r in "${ROBOTS[@]}"; do
        echo "robots_${r}.zip|robots/${r}"
    done
}

manifest_assets() {
    # assets/graspnet/**            -> assets_graspnet.zip
    echo "assets_graspnet.zip|assets/graspnet"
    # assets/objects/**             -> assets_objects.zip
    echo "assets_objects.zip|assets/objects"
    # assets/articulated/<id>/**    -> assets_articulated.zip
    echo "assets_articulated.zip|assets/articulated"
}

# assets/objaverse/<NNNN> -> assets_objaverse_NNNN.zip
# simple.assets.objaverse exposes ids 0..1544 (len_objaverse = 1545) and
# ObjaverseAssetManager.sample() can return any of them; the repo also carries 1545.
OBJAVERSE_MAX=1545

manifest_objaverse() {
    local i
    for ((i = 0; i <= OBJAVERSE_MAX; i++)); do
        printf 'assets_objaverse_%04d.zip|assets/objaverse/%04d\n' "$i" "$i"
    done
}

# scenes/hssd/<name> -> scenes_hssd_<name>.zip, for every scene the scene
# manager can sample out of hssd-scenes/config.yaml.
manifest_scenes() {
    if [[ ! -f "$SCENE_CONFIG" ]]; then
        echo "⚠️  Scene config not found: $SCENE_CONFIG" >&2
        return 1
    fi
    local name
    while IFS= read -r name; do
        [[ -z "$name" ]] && continue
        echo "scenes_hssd_${name}.zip|scenes/hssd/${name}"
    done < <(grep -oE '^[[:space:]]*name:[[:space:]]*"?[0-9_]+"?' "$SCENE_CONFIG" \
                 | sed -E 's/.*name:[[:space:]]*"?([0-9_]+)"?.*/\1/' | sort -u)
}

# vMaterials_2/**/*.mdl -> vMaterials_2.zip
manifest_materials() {
    echo "vMaterials_2.zip|vMaterials_2"
}

build_plan() {  # -> lines "<zip>|<dir>" for the selected groups
    local g
    for g in "${SEL_GROUPS[@]}"; do
        "manifest_${g}"
    done
}

# ============================================================================
# Plan
# ============================================================================

# A zip re-uploaded to HF gets its version bumped in data/_versions.json; an
# extracted copy whose data/.<zip>.version stamp is older counts as missing.
VERSIONS_FILE="$DATA_DIR/_versions.json"
expected_version() {
    local v
    v=$(grep -oE "\"$1\"[[:space:]]*:[[:space:]]*[0-9]+" "$VERSIONS_FILE" 2>/dev/null | grep -oE '[0-9]+$' | head -n1)
    echo "${v:-1}"
}
local_version() {
    local v
    v=$(tr -dc '0-9' 2>/dev/null < "$DATA_DIR/.$1.version")
    echo "${v:-1}"
}

PLAN=()
while IFS= read -r line; do
    [[ -n "$line" ]] && PLAN+=("$line")
done < <(build_plan)

if [[ ${#PLAN[@]} -eq 0 ]]; then
    echo "Nothing selected. Use --help for usage information."
    exit 0
fi

TODO_ZIPS=()
TODO_TARGETS=()
SKIPPED=0
for entry in "${PLAN[@]}"; do
    zip_name="${entry%%|*}"
    target="${entry##*|}"
    if ! $FORCE && [[ -d "$DATA_DIR/$target" ]] \
        && (( $(local_version "$zip_name") >= $(expected_version "$zip_name") )); then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    TODO_ZIPS+=("$zip_name")
    TODO_TARGETS+=("$target")
done

echo "📋 Groups:     ${SEL_GROUPS[*]}"
echo "   Repo:       ${HF_REPO} (${HF_REPO_TYPE})"
echo "   Data dir:   ${DATA_DIR}"
echo "   Planned:    ${#PLAN[@]} archives"
echo "   Present:    ${SKIPPED} already extracted (skipped)"
echo "   To fetch:   ${#TODO_ZIPS[@]}"
echo ""

if [[ ${#TODO_ZIPS[@]} -eq 0 ]]; then
    echo "✅ Everything is already in place. Nothing to download."
    exit 0
fi

if $LIST_ONLY; then
    write_list_file
    HF_LIST_FILE="$LIST_FILE" run_py <<'PY'
import os
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
from huggingface_hub import HfApi

with open(os.environ["HF_LIST_FILE"]) as fh:
    wanted = [l.strip() for l in fh if l.strip()]
api = HfApi(token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))
info = api.repo_info("USC-PSI-Lab/SIMPLE", repo_type="dataset", files_metadata=True)
sizes = {s.rfilename: (s.size or 0) for s in info.siblings}

total, missing = 0, []
for name in wanted:
    if name not in sizes:
        missing.append(name)
        continue
    total += sizes[name]
    print(f"  {sizes[name]/1e6:10.1f} MB  {name}")
print(f"\n  {'-'*10}")
print(f"  {total/1e9:10.2f} GB  total download ({len(wanted)-len(missing)} archives)")
if missing:
    print(f"\n  ⚠️  {len(missing)} archive(s) not found in the repo:")
    for m in missing:
        print(f"      {m}")
PY
    exit 0
fi

# The docker flow writes into data/ as root; extracting as a normal user then
# fails halfway through. Catch that before spending the bandwidth.
UNWRITABLE=()
for target in "${TODO_TARGETS[@]}"; do
    probe="$DATA_DIR/$target"
    while [[ ! -e "$probe" ]]; do probe="$(dirname -- "$probe")"; done
    [[ -w "$probe" ]] && continue
    _seen=false
    for u in "${UNWRITABLE[@]:-}"; do [[ "$u" == "$probe" ]] && _seen=true; done
    $_seen || UNWRITABLE+=("$probe")
done

if [[ ${#UNWRITABLE[@]} -gt 0 ]]; then
    echo "❌ Cannot extract into these paths (not writable by $(id -un)):" >&2
    printf '   - %s\n' "${UNWRITABLE[@]}" >&2
    echo "" >&2
    echo "   They were most likely created by the docker flow, which runs as root." >&2
    echo "   Either take ownership on the host:" >&2
    echo "       sudo chown -R $(id -u):$(id -g) \"$DATA_DIR\"" >&2
    echo "   or run this script inside the container." >&2
    exit 1
fi

if $DRY_RUN; then
    echo "Would download and extract into ${DATA_DIR}:"
    printf '  %s\n' "${TODO_ZIPS[@]}"
    exit 0
fi

# ============================================================================
# Download
# ============================================================================

echo "📥 Downloading ${#TODO_ZIPS[@]} archive(s) from ${HF_REPO} (${JOBS} workers)..."
write_list_file
HF_LOCAL_DIR="$STAGE_DIR" HF_JOBS="$JOBS" HF_LIST_FILE="$LIST_FILE" run_py <<'PY'
import os

try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

from huggingface_hub import snapshot_download

with open(os.environ["HF_LIST_FILE"]) as fh:
    patterns = [l.strip() for l in fh if l.strip()]
if not patterns:
    raise SystemExit("no archives requested")
snapshot_download(
    repo_id="USC-PSI-Lab/SIMPLE",
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir=os.environ["HF_LOCAL_DIR"],
    max_workers=int(os.environ.get("HF_JOBS", "8")),
    token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
)
PY
DOWNLOAD_RC=$?

if [[ $DOWNLOAD_RC -ne 0 ]]; then
    echo "⚠️  Download step exited with status ${DOWNLOAD_RC}; extracting whatever arrived."
fi

# ============================================================================
# Extract (all archives are rooted at data/, matching resolve_data_path)
# ============================================================================

echo ""
echo "📦 Extracting..."

EXTRACTED=0
FAILED=()
for i in "${!TODO_ZIPS[@]}"; do
    zip_name="${TODO_ZIPS[$i]}"
    target="${TODO_TARGETS[$i]}"
    zip_path="$STAGE_DIR/$zip_name"

    if [[ ! -f "$zip_path" ]]; then
        FAILED+=("$zip_name (not downloaded)")
        continue
    fi

    if unzip -q -o "$zip_path" -d "$DATA_DIR"; then
        EXTRACTED=$((EXTRACTED + 1))
        expected_version "$zip_name" > "$DATA_DIR/.$zip_name.version"
        $KEEP_ZIPS || rm -f "$zip_path"
        if [[ ! -d "$DATA_DIR/$target" ]]; then
            echo "  ⚠️  $zip_name extracted but $target is missing"
        fi
    else
        FAILED+=("$zip_name (bad archive)")
        echo "  ❌ Failed to extract $zip_name"
    fi
done

# huggingface_hub leaves its bookkeeping behind when local_dir is used
if $KEEP_ZIPS; then
    rm -rf "$STAGE_DIR/.cache" 2>/dev/null
else
    rm -rf "$STAGE_DIR" 2>/dev/null
fi

echo ""
echo "✅ Extracted ${EXTRACTED} archive(s) into ${DATA_DIR}/"
if $KEEP_ZIPS; then
    echo "💾 Zip files kept in ${STAGE_DIR}/"
else
    echo "🗑️  Zip files deleted after extraction (use --keep-zips to retain them)"
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "⚠️  ${#FAILED[@]} archive(s) did not land:"
    printf '   - %s\n' "${FAILED[@]}"
    exit 1
fi
