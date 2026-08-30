# Evaluation

Policy evaluation uses a **client-server** split: a Psi-0 inference server, and
the SIMPLE client running the simulation with the external SONIC controller.

## 1. Start the Psi-0 server

From the [Psi-0](https://github.com/physical-superintelligence-lab/Psi0)
workspace:

```bash
serve_psi0_sonic_http \
    --policy psi0 \
    --port 8014 \
    --ckpt-step=40000 \
    --run-dir=.runs/finetune/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223 \
    --rtc-mode test_time \
    --action-exec-horizon=24
```

Keep this terminal open for the duration of the evaluation.

### From Docker

Psi-0 ships the same server as the `serve-psi0-sonic-http` compose service, on
the official `ghcr.io/physical-superintelligence-lab/psi0` image (retagged
`psi:train`, or built locally with `docker compose build psi`):

```bash
export RUN=sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223

docker compose run --rm serve-psi0-sonic-http \
    --policy psi0 \
    --port 8014 \
    --ckpt-step 40000 \
    --run-dir .runs/finetune/$RUN \
    --rtc \
    --action-exec-horizon 24
```

`serve_psi0_sonic_http` is the service's entrypoint, so these arguments replace
its default argument list wholesale. `--run-dir` is relative to the container's
working directory, where the repo's `.runs/` is bind-mounted, so a checkpoint
downloaded on the host needs no path translation.

The service already defaults to that run at port 8014, step 40000, `--rtc` and
a horizon of 24, so

```bash
docker compose up serve-psi0-sonic-http
```

starts the same server; override single values from the shell or `.env`
instead of repeating the list — `POLICY`, `PORT`, `CKPT_STEP`, `RUN` (or
`RUN_DIR`), `RTC_FLAG`, `ACTION_EXEC_HORIZON`, and `GPUS` for the device.

The service is host-networked and binds `0.0.0.0:8014`, which is exactly what
`eval-sonic-wbc` below expects at `127.0.0.1:8014` — both containers share the
host network namespace.

## 2. Run the evaluation client

In a new terminal, from the SIMPLE repository root:

```bash
HEADLESS=0 bash scripts/run_eval_wbc.sh \
  simple/G1WholebodyXMoveBendCarryBoxSonic-v0 psi0 \
  --data-dir data/teleop_wbc/simple/G1WholebodyXMoveBendCarryBoxSonic-v0/level-0 \
  --host 127.0.0.1 \
  --port 8014 \
  --episode-start 1 \
  --num-episodes 1 \
  --dr-level 0 \
  --eval-dir /tmp/sonic-psi0-eval
```

The first two arguments are the environment id and the policy name; everything
after them is forwarded to the evaluation CLI. `--port` must match the server's
`--port`.

`HEADLESS=0` shows the desktop viewer and needs a live X display; use
`HEADLESS=1` to run without one — the evaluation still writes its MP4s.

The script launches the lockstep SONIC controller itself, waits for it to report
ready, and tears it down on exit. It refuses to start if ports 5556, 5557 or
13579 are already occupied, and requires the controller binary
(`third_party/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref`)
built by [Setup](setup.md).

### From Docker

The same client is wired up as the `eval-sonic-wbc` compose service. The
controller binary is already built into the image, and `network_mode: host` lets
the container reach a Psi-0 server running on the host at `127.0.0.1`:

```bash
docker compose run --rm eval-sonic-wbc \
  simple/G1WholebodyXMoveBendCarryBoxSonic-v0 psi0 \
  --data-dir data/teleop_wbc/simple/G1WholebodyXMoveBendCarryBoxSonic-v0/level-0 \
  --host 127.0.0.1 --port 8014 \
  --episode-start 1 --num-episodes 1 --dr-level 0 \
  --eval-dir data/eval/sonic-psi0
```

Keep `--data-dir` and `--eval-dir` under `data/` — that tree is bind-mounted, so
the results land on the host; a `/tmp` path dies with the container. The service
runs headless by default; add `-e HEADLESS=0` for the desktop viewer.

## 3. Results

Per-episode videos are written under `--eval-dir`, named
`episode_<id>/<cam_name>_<success_flag>.mp4`:

```bash
mpv /tmp/sonic-psi0-eval/psi0/G1WholebodyXMoveBendCarryBoxSonic-v0/level-0/episode_1/head_stereo_left_success.mp4
```
