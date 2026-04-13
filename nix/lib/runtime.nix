{ lib, pkgs }:

let
  exportEnv = env:
    lib.concatStringsSep "\n" (
      lib.mapAttrsToList (name: value: "export ${name}=${lib.escapeShellArg value}") env
    );

  prependEnv = env:
    lib.concatStringsSep "\n" (
      lib.mapAttrsToList (
        name: value:
          ''
            _runtime_current_value="$(printenv ${name} || true)"
            if [ -n "$_runtime_current_value" ]; then
              export ${name}=${lib.escapeShellArg value}:"$_runtime_current_value"
            else
              export ${name}=${lib.escapeShellArg value}
            fi
            unset _runtime_current_value
          ''
      ) env
    );

  mergeRuntimes = runtimes: {
    packages = lib.unique (lib.concatMap (runtime: runtime.packages or [ ]) runtimes);
    env = lib.foldl' (acc: runtime: acc // (runtime.env or { })) { } runtimes;
    prependEnv = lib.foldl' (acc: runtime: acc // (runtime.prependEnv or { })) { } runtimes;
    shellHook = lib.concatStringsSep "\n\n" (
      lib.filter (chunk: chunk != "") (map (runtime: runtime.shellHook or "") runtimes)
    );
  };

  mkRuntimeShell = {
    name,
    runtimes,
    extraPackages ? [ ],
    extraShellHook ? "",
    stdenv ? null,
  }:
    let
      runtime = mergeRuntimes runtimes;
      shellHook = lib.concatStringsSep "\n\n" (
        [
          (exportEnv (runtime.env or { }))
          (prependEnv (runtime.prependEnv or { }))
          (runtime.shellHook or "")
          extraShellHook
        ]
      );
    in
    pkgs.mkShell ({
      packages = lib.unique (extraPackages ++ runtime.packages);
      inherit shellHook;
    } // lib.optionalAttrs (stdenv != null) { inherit stdenv; });
in
{
  inherit exportEnv prependEnv mergeRuntimes mkRuntimeShell;
}
