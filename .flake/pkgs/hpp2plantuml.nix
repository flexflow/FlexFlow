{buildPythonPackage, fetchPypi}:

buildPythonPackage rec {
  pname = "hpp2plantuml";
  version = "0.8.5";
  format = "wheel";
  src = fetchPypi {
    inherit pname version format;
    sha256 = "sha256-PfTJmBypI21AAK3sMojygQfrhnRqcMmVCW4dxGfDfQg=";
  };
}
