{ buildPythonPackage
, fetchPypi
, jinja2
, robotpy-cppheaderparser
, sphinx
}:

buildPythonPackage rec {
  pname = "hpp2plantuml";
  version = "0.8.5";
  format = "wheel";
  src = fetchPypi {
    inherit pname version format;
    sha256 = "sha256-PfTJmBypI21AAK3sMojygQfrhnRqcMmVCW4dxGfDfQg=";
  };

  # argparse is part of the python 3 standard library, so there is no
  # corresponding nixpkgs package to depend on.
  pythonRemoveDeps = [ "argparse" ];

  dependencies = [
    jinja2
    robotpy-cppheaderparser
    sphinx
  ];

  pythonImportsCheck = [ "hpp2plantuml" ];
}
