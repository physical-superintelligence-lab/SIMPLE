# Setup

## Platform

Linux x86_64 with an NVIDIA GPU + CUDA. The SONIC controller (TensorRT/ONNX) and
curobo are built from source against CUDA; macOS/ARM is unsupported.

## Install

1. **Standard SIMPLE install** — follow the README **[Option 1] UV setup**:

   ```bash
   git submodule update --init --recursive
   uv sync --all-groups --index-strategy unsafe-best-match
   bash scripts/install_curobo.sh
   ```

2. **WBC extras** — one script adds what the standard install does not: the
   XRoboToolkit VR SDK build, the GR00T-WholeBodyControl controller and SONIC
   checkpoints, and the SONIC C++/TensorRT controller binary (it calls
   `build_sonic_controller.sh` as its last step).

   ```bash
   bash scripts/setup_teleop_wbc.sh
   ```

3. **XRoboToolkit PC Service** installed and running on the workstation wired to
   the headset.

### Docker

The image built by the repo `Dockerfile` performs steps 1-2 for you, including
the SONIC C++/TensorRT controller (`SIMPLE_BUILD_SONIC_WBC=1`, the default) and
the SONIC ONNX checkpoints (`SIMPLE_DOWNLOAD_SONIC_CKPT=1`). Docker cannot fetch
git submodules or git-LFS blobs itself, so populate them on the host first:

```bash
git submodule update --init --depth 1 third_party/GR00T-WholeBodyControl
git -C third_party/GR00T-WholeBodyControl lfs install --local
git -C third_party/GR00T-WholeBodyControl lfs pull --include="gear_sonic_deploy/**"
docker compose build sim          # add SIMPLE_FULL_INSTALL=1 for real-robot / GUI teleop
```

The controller binary lands at its usual path
(`third_party/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref`)
and its runtime library paths are in `~/tools/sonic_env.sh`, which the
`scripts/run_*_wbc.sh` launchers source — so inside the container the launch
commands are unchanged. The XRoboToolkit **PC Service** still runs on the host,
not in the container.

## Required hardware

* **PICO 4 / PICO 4 Ultra** headset
* **2x PICO controllers**
* **2x PICO motion trackers** — strapped to the left and right ankles, indicator
  lights facing upward

## Headset configuration

1. Pair and calibrate the motion trackers in the PICO system settings, in
   **Full-body** mode.
2. Wear tight-fitting trousers — baggy clothing blocks tracker line-of-sight and
   causes erratic leg motion.
3. Open **XRoboToolkit** on the headset, connect to your PC's IP address, and
   confirm the status reads **`Working`**.
4. In XRoboToolkit settings:
   * Tracking: **Head** and **Controller**
   * Motion Tracker: **Full-body**
   * Remote Vision: **Zedmini** (Listen enabled for visual feedback)

For step-by-step screenshots of the PICO pairing, app install and calibration
flow, see [Teleop Setup](../decoupled-wbc/teleop_decoupled_setup.md) — the
headset-side setup is shared between the two whole-body paths.
