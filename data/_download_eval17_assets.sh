#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
#
# Pre-download everything the simulator loads when evaluating the 17 built-in G1
# whole-body tasks (12 *Teleop-v0 + 5 *MP-v0, see docs/eval_decoupled17.md), so the
# evals run without network access.  Only what those evals touch is fetched (~6.6 GB),
# not the full ~27 GB asset set (use data/_download.sh for that).
#
#   bash data/_download_eval17_assets.sh            # download + extract + verify (in docker)
#   bash data/_download_eval17_assets.sh --list     # show the plan with sizes
#   bash data/_download_eval17_assets.sh --verify   # only check that everything is in place
#   bash data/_download_eval17_assets.sh --host     # run on the host instead of in docker
#
# What the 17 evals load (from scanning every episode's environment_config in
# data/SIMPLE-eval/<Task>/<level>/meta/episodes.jsonl and the task sources):
#
#   USC-PSI-Lab/SIMPLE (dataset)            extracted to         used by
#   --------------------------------------- -------------------- -------------------------------
#   robots_g1.zip                           data/robots/g1       g1_wholebody (MP) + g1_sonic USD
#   robots_g1_sonic.zip                     data/robots/g1_sonic g1_sonic MJCF (Teleop)
#   assets_graspnet.zip                     data/assets/graspnet targets / distractors
#   assets_objects.zip                      data/assets/objects  containers
#   assets_articulated.zip                  data/assets/articulated  door / oven / faucet / ...
#   vMaterials_2.zip                        data/vMaterials_2    table / ground material DR
#   scenes_hssd_<name>.zip (8 scenes)       data/scenes/hssd/<name>
#
#   USC-PSI-Lab/SIMPLE-eval (dataset)       copied to
#   <Task>/dr-level-* (17 tasks, unzipped)  data/SIMPLE-eval/<Task>/dr-level-*
#
# No Objaverse objects are used.  Not covered here (already in the docker image or the
# repo): the decoupled-WBC / SONIC controller and its ONNX checkpoints (baked into the
# image), the AMO lower-body policy (src/simple/robots/policy/*.pt), and the psi0 policy
# checkpoint (served by the separate policy server).
#
# For offline runs also set HF_HUB_OFFLINE=1 so nothing tries to reach the Hub.
#
# The default runner is the `sim` docker container (it has huggingface_hub, and data/
# is usually root-owned from earlier docker runs); --host runs with a local Python that
# has huggingface_hub (set PYTHON=..., or uv is used).  Idempotent: an archive whose
# extracted directory already exists, or a task whose eval levels are all present, is
# skipped.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_REL="$(realpath --relative-to="$ROOT_DIR" "$SCRIPT_PATH")"

usage() {
  cat <<EOF
Usage: $SCRIPT_REL [--list | --verify] [--host] [--force] [--keep-zips] [--jobs N]

  --list        print the archives still to fetch, with sizes, and exit
  --verify      only check that every file the 17 evals reference is present
  --host        run on the host (needs a Python with huggingface_hub) instead of docker
  --force       re-download archives / eval sets that are already in place
  --keep-zips   keep downloaded zips and eval files in data/.predownload-eval17/
  --jobs N      parallel downloads (default 8)

Env: DATE (docker image tag simple:\$DATE, default latest), DATA_DIR (host data dir,
default ./data), HF_TOKEN, PYTHON (host mode).
EOF
}

MODE=download RUNNER=docker IN_CONTAINER=0 FORCE=0 KEEP_ZIPS=0 JOBS=8
ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) MODE=list; shift ;;
    --verify) MODE=verify; shift ;;
    --host) RUNNER=host; shift ;;
    --in-container) IN_CONTAINER=1; shift ;;
    --force) FORCE=1; shift ;;
    --keep-zips) KEEP_ZIPS=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --jobs=*) JOBS="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Docker runner: re-run this script inside the `sim` container (data/ is bind-mounted
# there, see docker-compose.yml).  The script itself is mounted explicitly, so this
# also works when DATA_DIR points somewhere other than the data/ dir holding it.
# ---------------------------------------------------------------------------
if [[ "$RUNNER" == docker && "$IN_CONTAINER" == 0 ]]; then
  cd "$ROOT_DIR"
  mkdir -p "${DATA_DIR:-./data}"
  exec docker compose run --rm -T -v "$SCRIPT_PATH:/tmp/download_eval17_assets.sh:ro" \
    sim /tmp/download_eval17_assets.sh --in-container "${ARGS[@]}"
fi

if [[ "$IN_CONTAINER" == 1 ]]; then
  DATA="/workspace/simple/data"
  PY=/workspace/simple/.venv/bin/python
  PY_CMD=("$PY")
else
  DATA="$(realpath -m "${DATA_DIR:-$ROOT_DIR/data}")"
  if [[ -n "${PYTHON:-}" ]]; then PY_CMD=("$PYTHON")
  elif "$ROOT_DIR/.venv/bin/python" -c "import huggingface_hub" 2>/dev/null; then PY_CMD=("$ROOT_DIR/.venv/bin/python")
  elif python3 -c "import huggingface_hub" 2>/dev/null; then PY_CMD=(python3)
  elif command -v uv >/dev/null; then PY_CMD=(uv run --no-project --quiet --with huggingface_hub python)
  else echo "No Python with huggingface_hub found; set PYTHON=... or use the docker runner." >&2; exit 1
  fi
fi
SCENE_CONFIG="$ROOT_DIR/src/simple/resources/hssd-scenes/config.yaml"
[[ "$IN_CONTAINER" == 1 ]] && SCENE_CONFIG=/workspace/simple/src/simple/resources/hssd-scenes/config.yaml

export E17_DATA="$DATA" E17_MODE="$MODE" E17_FORCE="$FORCE" E17_KEEP_ZIPS="$KEEP_ZIPS" \
  E17_JOBS="$JOBS" E17_SCENE_CONFIG="$SCENE_CONFIG"

exec "${PY_CMD[@]}" - <<'PY'
import glob, json, os, re, shutil, sys, zipfile

DATA = os.environ["E17_DATA"]
MODE = os.environ["E17_MODE"]
FORCE = os.environ["E17_FORCE"] == "1"
KEEP_ZIPS = os.environ["E17_KEEP_ZIPS"] == "1"
JOBS = int(os.environ["E17_JOBS"])
STAGE = os.path.join(DATA, ".predownload-eval17")
ASSET_REPO = "USC-PSI-Lab/SIMPLE"        # zipped assets, extracted into data/
EVAL_REPO = "USC-PSI-Lab/SIMPLE-eval"    # unzipped eval sets: <Task>/dr-level-*/
EVAL_ROOT = os.path.join(DATA, "SIMPLE-eval")
LEVELS = ["dr-level-0", "dr-level-1", "dr-level-2", "dr-level-hard"]

TASKS = [
    # Teleop (decoupled WBC, robot g1_sonic)
    "G1WholebodyBendHandoverTeleop-v0", "G1WholebodyBendPickAndPlaceTeleop-v0",
    "G1WholebodyBendPickTeleop-v0", "G1WholebodyCloseDoorTeleop-v0",
    "G1WholebodyHandoverTeleop-v0", "G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
    "G1WholebodyOpenFaucetTeleop-v0", "G1WholebodyOpenOvenTeleop-v0",
    "G1WholebodyOpenTrashCanTeleop-v0", "G1WholebodyPickAndPlaceAndHugContainerTeleop-v0",
    "G1WholebodyPushOfficeChairTeleop-v0", "G1WholebodyXMovePickTeleop-v0",
    # MP (AMO lower body, robot g1_wholebody)
    "G1WholebodyBendPickAndPlaceOnSofaMP-v0", "G1WholebodyBendPickMP-v0",
    "G1WholebodyPickAndBendPlaceMP-v0", "G1WholebodyTabletopGraspMP-v0",
    "G1WholebodyTabletopHandoverMP-v0",
]

# HSSD scene uid -> scene dir name (src/simple/resources/hssd-scenes/config.yaml).
# scene1 and scene3 share one scene; uids used by the 17 tasks' room_choices:
#   scene0: BendHandover/BendPickAndPlace/BendPick(Teleop,MP)/CloseDoor/Handover/XMovePick
#   scene1: BendPickAndPlaceOnSofaMP, OpenOven    scene2: OpenFaucet, TabletopGraspMP
#   scene3: PushOfficeChair   scene4: OpenTrashCan   scene6: LocomotionPickBetweenTables
#   scene14: PickAndPlaceAndHugContainer  scene15: TabletopHandoverMP  scene19: PickAndBendPlaceMP
SCENES = {
    "scene0": "107734119_175999932", "scene1": "102344280", "scene2": "107733960_175999701",
    "scene3": "102344280", "scene4": "102344049", "scene6": "102344022",
    "scene14": "102344307", "scene15": "102344328", "scene19": "103997403_171030405",
}

# (repo, path in repo, target under DATA).  ASSET_REPO entries are zips extracted into
# DATA; EVAL_REPO entries are a task's directory, copied to SIMPLE-eval/<Task>.
ITEMS = [(ASSET_REPO, zf, target) for zf, target in [
    ("robots_g1.zip", "robots/g1"),
    ("robots_g1_sonic.zip", "robots/g1_sonic"),
    ("assets_graspnet.zip", "assets/graspnet"),
    ("assets_objects.zip", "assets/objects"),
    ("assets_articulated.zip", "assets/articulated"),
    ("vMaterials_2.zip", "vMaterials_2"),
]]
for name in sorted(set(SCENES.values())):
    ITEMS.append((ASSET_REPO, f"scenes_hssd_{name}.zip", f"scenes/hssd/{name}"))
for t in TASKS:
    ITEMS.append((EVAL_REPO, t, f"SIMPLE-eval/{t}"))

# Files the robot classes load directly (src/simple/robots/{g1_wholebody,g1_sonic}.py).
ROBOT_FILES = [
    "robots/g1/curobo/g1_29dof_with_dex3.yml", "robots/g1/curobo/dex3_right.yml",
    "robots/g1/g1_29dof_wholebody_dex3.xml", "robots/g1/g1_29dof_wholebody_dex3.usd",
    "robots/g1_sonic/g1_29dof_with_hand.xml",
]


def extracted(target):
    path = os.path.join(DATA, target)
    if target.startswith("SIMPLE-eval/"):  # a task's eval set is complete once every level is there
        return all(os.path.isfile(os.path.join(path, l, "meta", "info.json")) for l in LEVELS)
    return os.path.isdir(path)


def token():
    try:
        from dotenv import load_dotenv; load_dotenv()
    except Exception:
        pass
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None


def todo():
    return [it for it in ITEMS if FORCE or not extracted(it[2])]


def pattern(repo, path):
    """snapshot_download allow-pattern for an item: the zip, or the task's whole dir."""
    return path if repo == ASSET_REPO else f"{path}/*"


def staged(repo, path):
    return os.path.join(STAGE, repo.replace("/", "__"), path)


def list_plan():
    from huggingface_hub import HfApi
    api, total = HfApi(token=token()), 0
    pending = todo()
    files = {}  # repo -> {path: size}
    for repo in sorted({it[0] for it in pending}):
        info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
        files[repo] = {s.rfilename: s.size or 0 for s in info.siblings}
    print(f"{len(ITEMS) - len(pending)} of {len(ITEMS)} items already in place under {DATA}")
    for repo, path, target in pending:
        if repo == ASSET_REPO:
            size = files[repo].get(path)
        else:
            sizes = [v for k, v in files[repo].items() if k.startswith(path + "/")]
            size = sum(sizes) if sizes else None
        total += size or 0
        print(f"  {'MISSING' if size is None else f'{size / 1e6:9.1f} MB'}  {repo}  {path}  -> {target}")
    print(f"  total download: {total / 1e9:.2f} GB")


def download():
    from huggingface_hub import snapshot_download
    pending = todo()
    print(f"{len(ITEMS) - len(pending)} of {len(ITEMS)} items already in place; fetching {len(pending)}")
    if not pending:
        return True
    os.makedirs(STAGE, exist_ok=True)
    ok = True
    # Stage downloads under STAGE rather than straight into data/, so a root-owned
    # data/.cache from earlier docker runs cannot break the hub's lock files.
    for repo in sorted({it[0] for it in pending}):
        pats = [pattern(r, p) for r, p, _ in pending if r == repo]
        print(f"downloading {len(pats)} item(s) from {repo} ...", flush=True)
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=pats,
                              local_dir=os.path.join(STAGE, repo.replace("/", "__")),
                              max_workers=JOBS, token=token())
        except Exception as exc:  # install whatever arrived, report the rest
            print(f"  download from {repo} failed: {exc}", file=sys.stderr)
            ok = False
    for repo, path, target in pending:
        src, dest = staged(repo, path), os.path.join(DATA, target)
        if repo == ASSET_REPO:
            if not os.path.isfile(src):
                print(f"  not downloaded: {repo} {path}", file=sys.stderr); ok = False; continue
            try:
                with zipfile.ZipFile(src) as z:
                    z.extractall(DATA)
            except zipfile.BadZipFile:
                print(f"  bad archive: {path}", file=sys.stderr); ok = False; continue
            verb = "extracted"
        else:
            if not os.path.isdir(src):
                print(f"  not downloaded: {repo} {path}/", file=sys.stderr); ok = False; continue
            if FORCE:  # drop stale files from an earlier copy
                shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            verb = "copied"
        print(f"  {verb} {path} -> {target}" + ("" if extracted(target) else "  (WARNING: target still incomplete)"))
        if not KEEP_ZIPS:
            shutil.rmtree(src) if os.path.isdir(src) else os.remove(src)
    if not KEEP_ZIPS:
        shutil.rmtree(STAGE, ignore_errors=True)
    return ok


def verify():
    """Check every file the 17 tasks' eval episodes reference, plus the robot files."""
    scene_names = dict(SCENES)
    cfg = os.environ.get("E17_SCENE_CONFIG", "")
    if os.path.isfile(cfg):  # prefer the live mapping if the source tree is around
        text = open(cfg).read()
        scene_names.update(re.findall(r'-\s*uid:\s*"?(\w+)"?\s*\n(?:.*\n)*?\s*name:\s*"?([\w]+)"?', text))
    need, missing_sets = set(ROBOT_FILES), []
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        elif isinstance(o, str):
            m = re.search(r"(?:^|/)data/((?:assets|robots|scenes|vMaterials_2)/.+\.(?:usd|xml|mdl|yml))$", o)
            if m:
                need.add(m.group(1))
            elif re.fullmatch(r"vMaterials_2/.+\.mdl", o):
                need.add(o)
    episodes = 0
    for t in TASKS:
        for lvl in LEVELS:
            ep = os.path.join(EVAL_ROOT, t, lvl, "meta", "episodes.jsonl")
            if not os.path.isfile(ep):
                missing_sets.append(f"SIMPLE-eval/{t}/{lvl}"); continue
            for line in open(ep):
                if not line.strip():
                    continue
                conf = json.loads(json.loads(line)["environment_config"])
                episodes += 1
                walk(conf)
                uid = conf["dr_state_dict"]["scene"]["uid"].split(":")[-1]
                name = scene_names.get(uid)
                need.add(f"scenes/hssd/{name}/{name}.usd" if name else f"scenes/hssd/<unknown uid {uid}>")
    missing = sorted(p for p in need if not os.path.exists(os.path.join(DATA, p)))
    print(f"verify: {episodes} eval episodes over {len(TASKS)} tasks x {len(LEVELS)} levels, "
          f"{len(need)} referenced files under {DATA}")
    for m in missing_sets:
        print(f"  MISSING eval set: {m}")
    for m in missing:
        print(f"  MISSING: {m}")
    if missing or missing_sets:
        print(f"verify: FAILED ({len(missing_sets)} eval sets, {len(missing)} files missing)")
        return False
    print("verify: OK - all assets for offline eval of the 17 tasks are in place")
    return True


if MODE == "list":
    list_plan()
elif MODE == "verify":
    sys.exit(0 if verify() else 1)
else:
    ok = download()
    sys.exit(0 if verify() and ok else 1)
PY
