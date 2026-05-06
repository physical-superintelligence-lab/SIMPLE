{
  nixConfig = {
    substituters = ["https://cache.nixos.org"];
    extra-substituters = [
      "https://nixpkgs-python.cachix.org"
      "https://ros.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nixpkgs-python.cachix.org-1:hxjI7pFxTyuTHn2NkvWCrAUcNZLNS3ZAvfYNuYifcEU="
      "ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo="
    ];
  };

  inputs.robo-nix.url = "github:ausbxuse/robo-nix/develop";

  # NOTE: generated plumbing. Most users should edit robo.nix,
  # pyproject.toml, and .python-version instead of this file.
  outputs = {robo-nix, ...}:
    robo-nix.lib.mkProjectFlakeFromManifest ./robo.nix;
}
