{ pkgs }:

let
  runtimeLib = import ./lib/runtime.nix {
    lib = pkgs.lib;
    inherit pkgs;
  };
  baseRuntime = import ./runtime-base.nix { inherit pkgs; };
  hostGpuRuntime = import ./runtime-host-gpu.nix { inherit pkgs; };
in
runtimeLib.mergeRuntimes [
  baseRuntime
  hostGpuRuntime
]
