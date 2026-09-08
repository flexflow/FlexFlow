{ fetchFromGitHub
, stdenv
, cmake
, pkg-config
, libclang
, libllvm
, lib
, zlib
, argparse
, nlohmann_json
, fmt
}:

stdenv.mkDerivation rec {
  pname = "fccf";
  version = "03d373fc65e2d7ceeac441ba4bbddfdc25618dff";

  src = fetchFromGitHub {
    owner = "p-ranav";
    repo = "fccf";
    rev = version;
    sha256 = "sha256-3NdPon5ZfjoGFFgBlb0rzRnfWgSopvAc5Gls2NWHaOE=";
  };

  nativeBuildInputs = [
    cmake
    pkg-config
  ];

  buildInputs = [
    libclang
    libllvm
    zlib
    argparse
    nlohmann_json
    fmt
  ];

  patches = [
    ./json-package-name.patch
    ./fix-argparse-include.patch
    ./fix-cstdint-include.patch
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS"
  ];

  meta = with lib; {
    description = "A command-line tool that quickly searches through C/C++ source code in a directory based on a search string and prints relevant code snippets that match the query";
    homepage = "https://github.com/p-ranav/fccf";
    license = licenses.mit;
  };
}
