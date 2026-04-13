# Troubleshooting

1. `uv run eval ...` or `python -c 'import torch'` fails inside `nix develop` with:

    > ImportError: libstdc++.so.6: cannot open shared object file: No such file or directory

    **Cause**
    A previous version of the Nix dev shell only prepended CUDA paths to `LD_LIBRARY_PATH`. Fresh shells could then miss the rest of the Nix userspace runtime, including `libstdc++.so.6`, Python shared libraries, FFmpeg, and related native dependencies.

    **Fix**
    Update to the current dev shell and reload it:
    ```bash
    cd ~/SIMPLE
    direnv reload
    ```

    Verify the full runtime is present:
    ```bash
    uv run python - <<'PY'
    import torch
    print(torch.__version__)
    PY
    ```

    If you need to run `nix` from inside an already-loaded dev shell, prefer:
    ```bash
    env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop
    ```
    This avoids making `nix` itself load the project runtime libraries.

    If this is a brand new host, run:
    ```bash
    ./scripts/nix/prereq-check.sh
    ```
    before debugging further.

1. `nix develop` itself fails before entering the shell with:

    > nix: .../libstdc++.so.6: version `CXXABI_1.3.15' not found

    **Cause**
    This is usually not a project dependency problem. It means the outer shell is exporting a polluted `LD_LIBRARY_PATH`, so the `nix` binary is loading the wrong `libstdc++.so.6` before it can even enter the dev shell.

    **Fix**
    Run `nix` with `LD_LIBRARY_PATH` cleared:
    ```bash
    env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop
    ```

    For single-command entry:
    ```bash
    env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop -c <command>
    ```

    To confirm the diagnosis:
    ```bash
    echo "$LD_LIBRARY_PATH"
    env -u LD_LIBRARY_PATH nix --version
    ```

    If clearing `LD_LIBRARY_PATH` fixes it, open a fresh shell or remove the global export from your shell startup files before re-entering the repo.

1. Isaac Sim fails to initialize the GPU or `vulkaninfo` shows:

    > vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER
    >
    > Failed to create any GPU devices
    >
    > libGLX_nvidia.so.0: cannot open shared object file

    **Cause**
    The host Vulkan ICD JSON points to NVIDIA driver libraries such as `libGLX_nvidia.so.0`, but those driver libraries are not visible inside the dev shell runtime.

    **Fix**
    The dev shell now stages required host NVIDIA driver libraries into `.runtime-state/host-libcuda` and prepends only that shim directory to `LD_LIBRARY_PATH`.
    Reload the shell:
    ```bash
    cd ~/SIMPLE
    direnv reload
    ```

    Verify Vulkan inside the dev shell:
    ```bash
    vulkaninfo --summary
    ```

    Expected result:
    - Vulkan instance creation succeeds
    - the NVIDIA GPU appears in the device list

    If this still fails after reload, check that the host actually provides the NVIDIA driver libraries:
    ```bash
    ls -l /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0
    ls -l /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1
    ```

1. `uv sync` failed with error:

    >   × Failed to download `isaacsim-cortex==4.5.0.0`
      ├─▶ Failed to extract archive: isaacsim_cortex-4.5.0.0-cp310-none-manylinux_2_34_x86_64.whl
      ├─▶ I/O operation failed during extraction
      ╰─▶ Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT (current value:30s).
      help: `isaacsim-cortex` (v4.5.0.0) was included because `simple` (v0.1.0) depends on `isaacsim[all]`
            (v4.5.0.0) which depends on `isaacsim-cortex

    **Solution** set `uv` timeout and `sync` again.
    ```
    export UV_HTTP_TIMEOUT=300
    uv sync
    ```

1. The dev shell exits early with:

    > [simple-dev] error: LD_LIBRARY_PATH contains non-runtime paths: ...
    >
    > [bootstrap] PYTHONPATH must be unset inside the Nix runtime

    **Cause**
    The Nix runtime now enforces a strict boundary. Only `/nix/store` userspace and the required host NVIDIA driver paths are allowed. Inherited host Python overrides and injected loader state are rejected because they cause non-reproducible host-specific breakage.

    **Fix**
    Remove the leaking variables from your outer shell and reload:
    ```bash
    unset PYTHONPATH PYTHONHOME LD_PRELOAD
    direnv reload
    ```

    To inspect the bad runtime entries before reload:
    ```bash
    printf '%s\n' "${LD_LIBRARY_PATH:-}" | tr ':' '\n'
    ```

1. `curobo` installtion error:
   
    > RuntimeError:
          The detected CUDA version (11.8) mismatches the version that was used to compile
          PyTorch (12.8). Please make sure to use the same CUDA versions.

    **Solution** Install [`cuda`](https://developer.nvidia.com/cuda-12-8-0-download-archive) verison 12.8.
    ```
    wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
    sudo sh cuda_12.8.0_570.86.10_linux.run
    ```

2. `curobo` installation error:

    > RuntimeError: CUDA error: misaligned address
        CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
        For debugging consider passing CUDA_LAUNCH_BLOCKING=1

    **Solution** try install curobo again.
    ```
    TORCH_CUDA_ARCH_LIST=8.9+PTX uv pip install --no-build-isolation -e ".[isaacsim]"
    ```

3. `uv pip install torch==2.7.0 --extra-index-url https://pypi.nvidia.com`

    > ❯ UV_HTTP_CONCURRENCY=1  uv pip install torch==2.7.0 --extra-index-url https://pypi.nvidia.com
    Resolved 25 packages in 808ms
    × Failed to download `torch==2.7.0`
    ├─▶ Failed to write to the distribution cache
    ├─▶ error decoding response body
    ├─▶ request or response body error
    ├─▶ error reading a body from connection
    ╰─▶ connection reset

    **Solution**
    ```
    rm -rf ~/.cache/uv
    # install again
    ```

3. Errors starts `IsaacSim`

    > 2025-08-28 23:35:04 [8,352ms] [Error] [omni.ext.plugin] Could not load the dynamic library from /home/songlin/workspace/SIMPLE/.venv/lib/python3.10/site-packages/isaacsim/extscache/omni.iray.libs-0.0.0+d02c707b.lx64.r/bin/iray/libneuray.so. Error: libGLU.so.1: cannot open shared object file: No such file or directory (note that this may be caused by a dependent library)

    **Solution** Install `libglu`
    ```
    sudo apt-get install libglu1-mesa
    ```

4. GLFWError: (65537) b'The GLFW library is not initialized'
    > Usually, this happens on a headless server when running mujoco
    
    **Solution** Use `EGL`
    ```
    export MUJOCO_GL=egl
    ```

5. Fatal Python error: Segmentation fault when calling `world.reset()`

    > It happens when you run IsaacSim on a headless Server with headless set to False.

    **Solution** 
    ```
    # make sure headless is True.
    ```

6. Erorr install CuRobo

    > fatal error: Python.h: No such file or directory
         12 | #include <Python.h>
            |          ^~~~~~~~~~

    **Solution**
    ```
    sudo apt-get install python3-dev
    ```

7. ImportError: libgmpxx.so.4: cannot open shared object file: No such file or directory
   
    **Solution**
    ```
    sudo apt install libgmp-dev libgmp10
    ```

8. × No solution found when resolving dependencies for split (markers: python_full_version == '3.10.*' and sys_platform == 'darwin'): ╰─▶ Because only cmake==3.25.0 is available and lerobot==0.3.3 depends on cmake>=3.29.0.1, we can conclude that lerobot==0.3.3 cannot be used. And because simple:lerobot depends on lerobot==0.3.3 and your project requires simple:lerobot, we can conclude that your project's requirements are unsatisfiable.

    **Solution**
    ```
    uv sync --group lerobot --index-strategy unsafe-best-match
    ```

9. 2025-10-18 22:38:03 [33,607ms] [Error] [omni.kit.app._impl] [py stderr]: TypeError: array.__init__() got an unexpected keyword argument 'owner'
    
    **Solution**
    Downgrade warp-lang
    ```
    uv pip install "warp-lang==1.7.0" --index-strategy unsafe-best-match
    ```
