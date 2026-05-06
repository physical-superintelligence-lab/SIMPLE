{
  schemaVersion = 1;
  envName = "simple";
  description = "Minimal Python robotics environment with common CLI tooling and native build support.";
  components = [
    "base"
    "python-uv"
    "native-build"
    "cuda-toolkit"
    "isaac-sim"
    "x11-gl"
    "mujoco"
    "media"
    "linux-headers"
  ];
  pythonVersion = "3.10";
  supportedSystems = [
    "x86_64-linux"
    "aarch64-linux"
    "x86_64-darwin"
    "aarch64-darwin"
  ];
  workspaceRoot = ".";
  cudaWheelVersion = "12.8";

  requiredDirectories = [
    "third_party/gsnet"
    "third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
    "third_party/evdev"
    "third_party/AMO"
    "third_party/unitree_sdk2_python"
    "third_party/gear_sonic"
    "third_party/curobo"
    "third_party/openpi-client"
    "third_party/decoupled_wbc"
  ];

  provenance = {
    generatedBy = "robo init";
    profile = "minimal";
    componentReasons = [
      {
        name = "base";
        source = "manual config";
        reason = "selected with --components";
      }
      {
        name = "cuda-toolkit";
        source = "manual config";
        reason = "selected with --components";
      }
      {
        name = "isaac-sim";
        source = "manual config";
        reason = "selected with --components";
      }
      {
        name = "linux-headers";
        source = "manual config";
        reason = "selected with --components";
      }
      {
        name = "media";
        source = "manual config";
        reason = "selected with --components";
      }
      {
        name = "mujoco";
        source = "manual config";
        reason = "selected with --components";
      }
      {
        name = "native-build";
        source = "manual config";
        reason = "selected with --components";
      }
      {
        name = "python-uv";
        source = "manual config";
        reason = "selected with --components";
      }
      {
        name = "x11-gl";
        source = "manual config";
        reason = "selected with --components";
      }
    ];
    inferred = [
      "python 3.10: pyproject.toml requires-python"
      "cudaWheelVersion=12.8: inferred from uv.lock"
      "pyproject.toml uses MuJoCo/simulation packages"
      "LeRobot workflows commonly need media and graphics runtime libraries"
      "Isaac Sim Python wheels need host NVIDIA CUDA and graphics runtime support"
      "workspace contains Qt service paths"
    ];
  };
}
