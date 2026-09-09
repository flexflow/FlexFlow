{ lib
, stdenv
, fetchFromGitHub
, cmake
, cudaPackages ? { }
, zlib
, maxDim ? 5
}:

let
  inherit (cudaPackages) cudatoolkit;
in

stdenv.mkDerivation rec {
  pname = "realm";
  version = "2026-08-28";

  src = fetchFromGitHub {
    owner = "StanfordLegion";
    repo = "realm";
    rev = "15b3e9b68b65d698d237766d37bc36522b8ecf0b";
    sha256 = "sha256-HXNRKzOIUok8FAuhJF1sOW0+Hv7rRfRDoQNt+wAvOnc=";
  };

  nativeBuildInputs = [
    cmake
  ];

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DREALM_ENABLE_CUDA=ON"
    "-DREALM_ENABLE_PREALM=ON"
    "-DREALM_MAX_DIM=${toString maxDim}"
  ];

  buildInputs = [
    cudatoolkit
    zlib
  ];

  meta = with lib; {
    description = "Realm is a distributed, event–based tasking runtime for building high-performance applications that span clusters of CPUs, GPUs, and other accelerators";
    homepage = "https://legion.stanford.edu/realm";
    license = licenses.asl20;
  };
}
