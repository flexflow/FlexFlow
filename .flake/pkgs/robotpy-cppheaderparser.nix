{ buildPythonPackage
, fetchPypi
, setuptools
, ply
}:

buildPythonPackage rec {
  pname = "robotpy-cppheaderparser";
  version = "5.1.2";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-FdNQs5NYtFzbH+E4r578zg5jLBtYgwoboboJihdSaYs=";
  };

  build-system = [ setuptools ];

  dependencies = [ ply ];

  pythonImportsCheck = [ "CppHeaderParser" ];
}
