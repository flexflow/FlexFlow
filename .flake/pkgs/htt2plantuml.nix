{ lib, buildPythonPackage, fetchPypi, python }:

buildPythonPackage rec {
  pname = "htt2plantuml";
  version = "0.1.0";
  src = fetchPypi {
    inherit pname version;
    sha256 = "fdjaklfdsa"; # Replace with the correct sha256 hash
  };

  propagatedBuildInputs = [
    python.pkgs.six
  ];

  meta = with lib; {
    description = "HTTP to PlantUML converter";
    homepage = "https://github.com/thibaultmarin/hpp2plantuml";
  };
}
