{ pkgs }:

{
  shellHook = ''
    if [ -f "$PWD/scripts/nix/common.sh" ]; then
      # shellcheck disable=SC1091
      . "$PWD/scripts/nix/common.sh"
      simple_runtime_stage_host_driver_libs "$PWD"
    fi
  '';
}
