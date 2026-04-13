{
  description = "SIMPLE development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
    nixpkgs-unstable,
  }: let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];

    forAllSystems = f:
      nixpkgs.lib.genAttrs systems (
        system:
          f system
      );
  in {
    lib = forAllSystems (system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      unstablePkgs = import nixpkgs-unstable {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      runtimeLib = import ./nix/lib/runtime.nix {
        lib = pkgs.lib;
        inherit pkgs;
      };
      baseRuntime = import ./nix/runtime-base.nix {
        inherit pkgs;
        pythonPkg = pkgs.python310;
        cudaPkgs = unstablePkgs;
      };
      hostGpuRuntime = import ./nix/runtime-host-gpu.nix { inherit pkgs; };
    in {
      inherit baseRuntime hostGpuRuntime;
      runtime = runtimeLib.mergeRuntimes [
        baseRuntime
        hostGpuRuntime
      ];
    });

    devShells = forAllSystems (system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      unstablePkgs = import nixpkgs-unstable {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      lib = pkgs.lib;
      runtimeLib = import ./nix/lib/runtime.nix {
        inherit lib pkgs;
      };
      runtime = self.lib.${system}.runtime;

      devTools = with pkgs; [
        bashInteractive
        coreutils
        curl
        git
        git-lfs
        python310
        unstablePkgs.uv
        wget
      ];
    in {
      default = runtimeLib.mkRuntimeShell {
        name = "simple-dev";
        runtimes = [ runtime ];
        extraPackages = devTools;
        stdenv = pkgs.stdenv;
        extraShellHook = ''
          export UV_PYTHON="''${UV_PYTHON:-${pkgs.python310}/bin/python3.10}"
          export ROOT_DIR="$PWD"
          if [ -f "$ROOT_DIR/scripts/nix/common.sh" ]; then
            . "$ROOT_DIR/scripts/nix/common.sh"
          fi
          if [ -f "$ROOT_DIR/scripts/nix/shell-init.sh" ]; then
            . "$ROOT_DIR/scripts/nix/shell-init.sh"
          fi
        '';
      };
    });
  };
}
