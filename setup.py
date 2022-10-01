from setuptools import setup, find_packages
from pathlib import Path
import sys, os, subprocess
from setuptools.command.build_ext import build_ext

datadir = Path(__file__).parent / 'python/flexflow'
files = [str(p.relative_to(datadir)) for p in datadir.rglob('*.py')]
wdir = os.getcwd()
ncores = os.cpu_count()

class CMake_Build_Extension(build_ext):
  def run(self):
    os.makedirs("build",exist_ok=True)
    os.chdir("build")
    build_cmds = ["FF_BUILD_FROM_PYPI=ON ../config/config.linux", f"make -j {ncores-1}"]
    
    for command in build_cmds:
      print(f"Running {command}")
      subprocess.run(command, shell=True, check=True, capture_output=False,)
      os.chdir(wdir)
    build_ext.run(self)

setup(
  name='flexflow',
  version='1.0',
  description='FlexFlow Python package',
  url='https://github.com/flexflow/FlexFlow',
  license='Apache',
  packages=find_packages("python"),
  include_package_data=True,
  has_ext_modules=lambda: True,
  package_dir={'': "python"},
  package_data={'flexflow': files},
  zip_safe= False,
  install_requires=['numpy>=1.16',
                    'cffi>=1.11',
                    'qualname>=0.1',
                    'keras_preprocessing',
                    'Pillow',
                    'pybind11',
                    'ninja'
                    ],
  entry_points = {
          'console_scripts': ['flexflow_python=flexflow.driver:flexflow_driver'],
      },
  cmdclass={
      'build_ext': CMake_Build_Extension
  },
  classifiers=[
      'Programming Language :: Python :: 3.6',
      'License :: OSI Approved :: Apache Software License',
      'Operating System :: POSIX :: Linux',
      'Topic :: Software Development :: Libraries',
  ],
  python_requires='>=3.6',
)
