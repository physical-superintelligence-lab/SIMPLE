# Developer Notes

## Quaternion Convention

[Mujoco](https://mujoco.readthedocs.io/en/2.2.1/programming.html) 

`q = (w, x, y, z)` 

[IsaacSim](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/reference_conventions.html)

| API | Representation |
|-----|----------------|
| Isaac Sim Core | (QW, QX, QY, QZ) |
| USD | (QW, QX, QY, QZ) |
| PhysX | (QX, QY, QZ, QW) |
| Dynamic Control | (QX, QY, QZ, QW) |

[Transforms3d](https://matthew-brett.github.io/transforms3d/reference/transforms3d.quaternions.html)

Quaternions here consist of 4 values `w, x, y, z`, where w is the real (scalar) part, and x, y, z are the complex (vector) part.

## Command Line Argument Parser

we use [typer](https://typer.tiangolo.com/#use-typer-in-your-code) to manage command line arguments. It's simple and elegant.

Check it out `scripts/replay.py` for example usage:
```
import typer

def main(
    dataset_dir: Annotated[str, typer.Argument()] = "./output/renders",
    sim_mode: str = "mujoco_isaac",
    headless: bool = False,
    max_episode_steps: int = 50,
    save_dir: str = "./output",
):
    ...

if __name__ == "__main__":
    typer.run(main)
```

You can now inspect command line `arguments` (which are **REQUIRED** by default) and `options` (whic are **OPTIONAL** by default):
```
python scripts/replay.py --help
```
You'll get

```
Usage: replay.py [OPTIONS] [DATASET_DIR]

Arguments:
  [DATASET_DIR]  [default: ./output/renders]

Options:
  --sim-mode TEXT              [default: mujoco_isaac]
  --headless / --no-headless   [default: no-headless]
  --max-episode-steps INTEGER  [default: 50]
  --save-dir TEXT              [default: ./output]
  --help                       Show this message and exit.
```
Now pass the arguments:
```
python scripts/replay.py xxx --no-headless
```
## Lazy Downloading

We would like to maintain a `lazy-downloading` strategy for users and try to ***minimize*** the downloading overhead as much as possible. Specifically, developers are encouraged to implement `auto-downloading` of absoluately needed data or files for a specific script.

There are two helper function to be used:
1. `resolve_data_path` which resolve local file path in folder `data/` and it raise `FileNotFoundError` error if file is not present. 

2. catch the error and use `snapshot_download` to download files from our huggingface repo `USC-PSI-Lab/SIMPLE`

*For example*, developers can follow the steps shown in the `Graspnet1BAssetManager.load` function:

```
try: 
    # 1. resolve some local file path which is need to run the script
    collision_mesh_dir = resolve_data_path(f'{self.src_dir}/collision_models_mujoco/{asset_id_int:03d}/')
except FileNotFoundError:
    # 2. catch file not found error if files are not auto-downloaded
    from huggingface_hub import snapshot_download
    local_data_dir = resolve_data_path()
    # 3. Now auto download it using huggingface-hub
    snapshot_download(
        repo_id="USC-PSI-Lab/SIMPLE",
        allow_patterns=["assets.zip"],
        local_dir=local_data_dir,
        repo_type="dataset",
        # resume_download=True,
        token=os.environ.get("HF_TOKEN"),
    )
    # 4. unzip the downloaded zip file
    import zipfile
    zip_path = os.path.join(local_data_dir, "assets.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(local_data_dir)

    # 5. resolve the local file path again
    collision_mesh_dir = resolve_data_path(f'{self.src_dir}/collision_models_mujoco/{asset_id_int:03d}/')
```


## Upload data folder to `HuggingFace`

```
hf upload USC-PSI-Lab/SIMPLE . --repo-type=dataset
```

