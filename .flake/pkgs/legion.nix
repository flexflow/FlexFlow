{ lib
, fetchFromGitLab
, cmake
, config
, python3
, cudaPackages ? { }
, cudaCapabilities ? [ "60" "70" "80" "86" ]
, rocmPackages ? { }
, maxDim ? 5
, useCuda ? config.cudaSupport
, useRocm ? config.rocmSupport
, stdenv ? if useCuda then cudaPackages.backendStdenv else rocmPackages.llvm.rocmClangStdenv
}:

# from https://codeberg.org/Uli/nix-things/src/commit/776519e382c81b136c1d0b10d8c7b52b4acb9192/overlays/cq/python/libclang-python.nix

let 
  cmakeFlag = x: if x then "1" else "0";
  inherit (cudaPackages) cudatoolkit;
  inherit (lib)
    cmakeBool
    cmakeFeature
    optionals
    ;

  cudaBuildInputs = with cudaPackages; [
    cudatoolkit
  ];
  rocmBuildInputs = with rocmPackages; [
    clr
    rocthrust
    rocprim
    llvm.clang
  ];
in

stdenv.mkDerivation rec {
  pname = "legion_flexflow";
  version = "2024-03-13";

  src = fetchFromGitLab {
    owner = "StanfordLegion";
    repo = "legion";
    rev = "24e8c452341dea41427e0ce61e154d61715e6835";
    sha256 = "sha256-NjCSjphOIew/V24i74I6DModSGcWKLeiSIjts3cFtx4=";
    fetchSubmodules = true;
  };

  nativeBuildInputs = [
    cmake
  ];

  cmakeFlags = [
    "-DLegion_USE_Python=1"
    "-DLegion_BUILD_BINDINGS=1"
    "-DLegion_MAX_DIM=${toString maxDim}"
  ]
  ++ optionals useRocm [
        # TODO: this is the legacy way of setting hip compiler. Once we update nixpkgs version we should use the new way. It will be a quick fix
        (cmakeFeature "Legion_USE_HIP" "1")
        (cmakeFeature "HIP_ARCHITECTURES" (builtins.concatStringsSep ";" rocmPackages.clr.gpuTargets))
        (cmakeFeature "HIP_COMPILER" "${rocmPackages.llvm.clang}/bin/clang")
        (cmakeFeature "HIP_RUNTIME" "rocclr")
        (cmakeFeature "HIP_PLATFORM" "amd")
        (cmakeFeature "HIP_PATH" "${rocmPackages.clr}/hip")
        (cmakeFeature "HIP_ROOT_DIR" "${rocmPackages.clr}")
        (cmakeFeature "HIP_THRUST_ROOT_DIR" "${rocmPackages.rocthrust}")
        (cmakeFeature "ROCM_PATH" "${rocmPackages.clr}")

        (cmakeFeature "CMAKE_CXX_COMPILER" "hipcc")
        (cmakeFeature "CMAKE_C_COMPILER" "hipcc")
      ]
  ++ optionals useCuda [
        (cmakeFeature "Legion_USE_CUDA" "1")
        (cmakeFeature "CMAKE_CUDA_ARCH" (builtins.concatStringsSep ";" cudaCapabilities))
      ];



  buildInputs = [ 
    python3
  ]
  ++ optionals useCuda cudaBuildInputs
  ++ optionals useRocm rocmBuildInputs;

  meta = with lib; {
    description = "Legion is a parallel programming model for distributed, heterogeneous machines";
    homepage = "https://github.com/StanfordLegion/legion";
    license = licenses.asl20;
  };
}