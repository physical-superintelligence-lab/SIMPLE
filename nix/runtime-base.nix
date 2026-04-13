{ pkgs, pythonPkg ? null, cudaPkgs ? pkgs }:

let
  cudaPackages =
    if builtins.hasAttr "cudaPackages_12_8" cudaPkgs then cudaPkgs.cudaPackages_12_8
    else if builtins.hasAttr "cudaPackages_12_3" cudaPkgs then cudaPkgs.cudaPackages_12_3
    else if builtins.hasAttr "cudaPackages_12_2" cudaPkgs then cudaPkgs.cudaPackages_12_2
    else if builtins.hasAttr "cudaPackages_12_1" cudaPkgs then cudaPkgs.cudaPackages_12_1
    else if builtins.hasAttr "cudaPackages_12_0" cudaPkgs then cudaPkgs.cudaPackages_12_0
    else if builtins.hasAttr "cudaPackages_11_8" cudaPkgs then cudaPkgs.cudaPackages_11_8
    else throw "No supported cudaPackages set found in this nixpkgs revision";
  cuda = cudaPackages.cudatoolkit;
  stdenvWrapper = pkgs.stdenv.cc;
  stdenvCc = stdenvWrapper.cc;
  gcc = stdenvWrapper;
  gccRuntime =
    if builtins.hasAttr "lib" stdenvCc
    then stdenvCc.lib
    else stdenvWrapper.lib;
  libcDev =
    if builtins.hasAttr "libc_dev" stdenvWrapper
    then stdenvWrapper.libc_dev
    else pkgs.glibc.dev;
  ffmpeg = pkgs.ffmpeg_7;
  ffmpegLib = ffmpeg.lib;
  linuxHeaders = pkgs.linuxHeaders;
  xorg = pkgs.xorg or { };
  xorgLib = name: builtins.getAttr name xorg;
  pkgOrXorg = attr: legacy:
    if builtins.hasAttr attr pkgs
    then builtins.getAttr attr pkgs
    else xorgLib legacy;
  libxcryptRuntime =
    if builtins.hasAttr "libxcrypt-legacy" pkgs
    then pkgs.libxcrypt-legacy
    else pkgs.libxcrypt;
  pythonRuntime =
    if pythonPkg != null
    then pythonPkg
    else if builtins.hasAttr "python310" pkgs
    then pkgs.python310
    else throw "runtime-base.nix requires pythonPkg when pkgs has no python310";
  pybind11Pkg =
    if builtins.hasAttr "pybind11" pkgs
    then pkgs.pybind11
    else pythonRuntime.pkgs.pybind11;

  runtimeLibs = with pkgs; [
    pythonRuntime
    zlib
    glib
    gmp
    libxcryptRuntime
    ffmpegLib
    libglvnd
    mesa
    vulkan-loader
    libGL
    libGLU
    (pkgOrXorg "libx11" "libX11")
    (pkgOrXorg "libxcursor" "libXcursor")
    (pkgOrXorg "libxi" "libXi")
    (pkgOrXorg "libxrandr" "libXrandr")
    (pkgOrXorg "libxext" "libXext")
    (pkgOrXorg "libxrender" "libXrender")
    (pkgOrXorg "libxfixes" "libXfixes")
    (pkgOrXorg "libxinerama" "libXinerama")
    (pkgOrXorg "libxcb" "libxcb")
    (pkgOrXorg "libxau" "libXau")
    (pkgOrXorg "libxdmcp" "libXdmcp")
    (pkgOrXorg "libxxf86vm" "libXxf86vm")
    (pkgOrXorg "libsm" "libSM")
    (pkgOrXorg "libice" "libICE")
    (pkgOrXorg "libxt" "libXt")
    libxkbcommon
    dbus
    nspr
    nss
    expat
    fontconfig
    freetype
  ];

  runtimeLibPath = pkgs.lib.makeLibraryPath (runtimeLibs ++ [ gccRuntime ]);

in
{
  packages = runtimeLibs ++ [
    gccRuntime
    cuda
    pkgs.cmake
    gcc
    linuxHeaders
    ffmpeg
    pkgs.gnumake
    pkgs.ninja
    pkgs.patchelf
    pybind11Pkg
    pkgs.pkg-config
    pkgs.vulkan-tools
  ];

  env = {
    SIMPLE_LINUX_HEADERS = "${linuxHeaders}/include";
    SIMPLE_LIBC_DEV = "${libcDev}";
    OMNI_KIT_ACCEPT_EULA = "Y";
    MUJOCO_GL = "egl";
    PYOPENGL_PLATFORM = "egl";
    CUDA_HOME = "${cuda}";
    CUDA_PATH = "${cuda}";
    CC = "${gcc}/bin/cc";
    CXX = "${gcc}/bin/c++";
    CUDAHOSTCXX = "${gcc}/bin/c++";
    CMAKE_CUDA_HOST_COMPILER = "${gcc}/bin/cc";
    CMAKE_INCLUDE_PATH = "${linuxHeaders}/include:${cuda}/include";
    NVCC_PREPEND_FLAGS = "-ccbin ${gcc}/bin/cc";
  };

  prependEnv = {
    LD_LIBRARY_PATH = "${cuda}/lib:${cuda}/lib64:${runtimeLibPath}";
    CPATH = "${linuxHeaders}/include:${cuda}/include";
    C_INCLUDE_PATH = "${linuxHeaders}/include:${cuda}/include";
    CPLUS_INCLUDE_PATH = "${linuxHeaders}/include:${cuda}/include";
    LIBRARY_PATH = "${cuda}/lib64:${cuda}/lib";
  };

  shellHook = ''
    export CFLAGS="-I${linuxHeaders}/include ''${CFLAGS:-}"
    export CXXFLAGS="-I${linuxHeaders}/include ''${CXXFLAGS:-}"
  '';
}
