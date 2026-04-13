# Policy Evaluation

Benchmark tasks for $\Psi_0$

| Tasks | Whole-body | Motion Planning | Teleoperation |
|----------|----------|----------|----------| 
| G1WholebodyBendPickMP-v0 | x  | v | x    
| G1WholebodyHandoverTeleop-v0  | x  | v  | v
| G1WholebodyLocomotionPickBetweenTablesTeleop-v0  | v | x  | v
| G1WholebodyTabletopGraspMP-v0  | x  | v  | x
| G1WholebodyXMoveBendPickTeleop-v0  | v  | x | v
| G1WholebodyXMovePickTeleop-v0  | v  | x  | v


## Download eval data

```
export task=G1WholebodyBendPickMP-v0
```

```
hf download USC-PSI-Lab/psi-data \
    simple-eval/$task.zip \
    --local-dir=data/evals \
    --repo-type=dataset

unzip data/evals/simple-eval/$task.zip -d data/evals/simple-eval
```

## Start Server

> Make sure you start the policy server first. If you run SIMPLE locally on a workstation, we suggest start VLAs models on a different PC for better performance.

> If the server is started on a remote server, run ssh port forward. eg., ssh -L 22086:localhost:22086 songlin@nebula100.

> Once port forward is done, open a new terminal to test if server is up curl -i http://localhost:22085/health

## Run client

```
MUJOCO_GL=egl uv run eval simple/FrankaTabletopGrasp-v0 \
    openvla \
    --host=127.0.0.1 \
    --port=21075 \
    --sim-mode=mujoco_isaac \
    --headless \
    --max-episode-steps=50
```

or use docker, you can optionally set gpu device usig `GPUs={device_id}`
```
GPUs=1 docker compose run eval simple/FrankaTabletopGrasp-v0 \
    openvla \
    --host=172.17.0.1 \
    --port=21075 \
    --sim-mode=mujoco_isaac \
    --headless \
    --max-episode-steps=50
```

Find results at `./data/evals/openvla`