{ lib
, stdenv
, fetchFromGitHub
, cmake
, rustc
, cargo
}:

stdenv.mkDerivation rec {
  pname = "tokenizers-cpp";
  version = "2024-03-13";

  src = fetchFromGitHub {
    owner = "mlc-ai";
    repo = "tokenizers-cpp";
    rev = "4f42c9fa74946d70af86671a3804b6f2433e5dac";
    sha256 = "sha256-p7OYx9RVnKUAuMexy3WjW2zyfMJ/Q9ss4xFLsbQK7wA=";
    fetchSubmodules = true;
  };

  nativeBuildInputs = [
    cmake
    rustc
  ];

  # cmakeFlags = [
  #   "-DLegion_USE_Python=1"
  #   "-DLegion_BUILD_BINDINGS=1"
  #   "-DLegion_USE_CUDA=1"
  #   "-DLegion_CUDA_ARCH=${lib.concatStringsSep "," cudaCapabilities}"
  # ];

  buildInputs = [ ];
    # python3
    # cudatoolkit
  # ];

  meta = with lib; {
    description = "Universal cross-platform tokenizers binding to HF and sentencepiece";
    homepage = "https://github.com/mlc-ai/tokenizers-cpp";
    license = licenses.asl20;
  };
}
