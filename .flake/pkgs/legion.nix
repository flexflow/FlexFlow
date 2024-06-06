{ lib
, stdenv
, fetchFromGitLab
, cmake
, clang
, python3
# , cudaPackages ? { }
# , cudaCapabilities ? [ "60" "70" "80" "86" ]
, rocm
, rocmPackages
, maxDim ? 5
}:

# from https://codeberg.org/Uli/nix-things/src/commit/776519e382c81b136c1d0b10d8c7b52b4acb9192/overlays/cq/python/libclang-python.nix

let 
  cmakeFlag = x: if x then "1" else "0";

  # inherit (cudaPackages) cudatoolkit;
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
    "-DLegion_USE_HIP=1"
    "-DHIP_THRUST_ROOT_DIR=${rocm}/hip-thrust"

    "-DLegion_USE_CUDA=0"
    # "-DLegion_CUDA_ARCH=${lib.concatStringsSep "," cudaCapabilities}"
    "-DLegion_MAX_DIM=${toString maxDim}"

  ];

  preConfigure = ''
  echo "configuring Legion"
  echo "including rocm path"
  export ROCM_PATH=${rocm}
  export HIP_PATH=${rocm}/hip
  export HIP_THRUST_ROOT_DIR=${rocm}/hip-thrust
  echo "rocm path is $ROCM_PATH"
  echo "hip path is $HIP_PATH"
  echo "hip thrust path is $HIP_THRUST_ROOT_DIR"
  '';

  preUnpack = ''
  echo "Running pre-unpack steps..."
'';

  buildInputs = [ 
    python3
    rocm
  ];

  meta = with lib; {
    description = "Legion is a parallel programming model for distributed, heterogeneous machines";
    homepage = "https://github.com/StanfordLegion/legion";
    license = licenses.asl20;
  };
}
