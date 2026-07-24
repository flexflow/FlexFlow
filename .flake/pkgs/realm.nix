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
  version = "2026-07-01";

  src = fetchFromGitHub {
    owner = "StanfordLegion";
    repo = "realm";
    rev = "dba83452d4a211e3a0c0e5d4849c26ce7b56bba3";
    sha256 = "sha256-dLGmUgupDX0dDErZbuP5ipj2Cwf+rOhVGC/+angRyBY=";
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
