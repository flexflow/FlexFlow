{ stdenv
, lib
, fetchurl
, alsa-lib
, openssl
, zlib
, pulseaudio
, autoPatchelfHook
}:

stdenv.mkDerivation rec {
  pname = "bencher-cli";
  version = "0.4.33";

  system = "x86_64-linux";

  src = fetchurl {
    url = "https://github.com/bencherdev/bencher/releases/download/v${version}/bencher-v${version}-linux-x86-64";
    hash = "sha256-3q2ZGSqbcMaUcNMGJN+IsEP/+RlYHnsmiWdJ2oV2qmw=";
  };

  nativeBuildInputs = [
    autoPatchelfHook
  ];

  dontUnpack = true;

  installPhase = ''
    ls
    runHook preInstall
    install -m755 -D $src $out/bin/bencher
    runHook postInstall
  '';

  meta = with lib; {
    homepage = "https://bencher.dev/";
    platforms = platforms.linux;
  };
}
