{ lib
, stdenv
, fetchFromGitLab
, cmake
, python3
, cudaPackages ? { }
, cudaCapabilities ? [ "60" "70" "80" "86" ]
, maxDim ? 5
}:

# from https://codeberg.org/Uli/nix-things/src/commit/776519e382c81b136c1d0b10d8c7b52b4acb9192/overlays/cq/python/libclang-python.nix

let 
  cmakeFlag = x: if x then "1" else "0";

  inherit (cudaPackages) cudatoolkit;
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
    "-DLegion_USE_CUDA=1"
    "-DLegion_CUDA_ARCH=${lib.concatStringsSep "," cudaCapabilities}"
    "-DLegion_MAX_DIM=${toString maxDim}"
  ];

  buildInputs = [ 
    python3
    cudatoolkit
  ];

  meta = with lib; {
    description = "Legion is a parallel programming model for distributed, heterogeneous machines";
    homepage = "https://github.com/StanfordLegion/legion";
    license = licenses.asl20;
  };
}
