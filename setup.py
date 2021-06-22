from setuptools import setup, find_packages
from pathlib import Path
from cmake_build_extension import BuildExtension, CMakeExtension
import sys
# from setuptools.command.install import install
#
# global enable_nccl
# enable_nccl = '-DFF_USE_NCCL=OFF'
#
# class InstallCommand(install):
#   user_options = install.user_options + [
#     ('nccl', None, 'Enable NCCL'),
#     #('someval=', None, None) # an option that takes a value
#   ]
#
#   def initialize_options(self):
#     install.initialize_options(self)
#     self.nccl = 0
#     #self.someval = None
#
#   def finalize_options(self):
#     install.finalize_options(self)
#
#   def run(self):
#       if self.nccl != None:
#         global enable_nccl
#         enable_nccl = '-DFF_USE_NCCL=ON'
#         print(self.nccl, enable_nccl)
#        # assert 0
#       install.run(self)

datadir = Path(__file__).parent / 'python/flexflow'
files = [str(p.relative_to(datadir)) for p in datadir.rglob('*.py')]

cmdclass = dict()
cmdclass['build_ext'] = BuildExtension

setup(
  name='flexflow',
  version='1.0',
  description='FlexFlow Python package',
  url='https://github.com/flexflow/FlexFlow',
  license='Apache',
  packages=find_packages("python"),
  package_dir={'': "python"},
  package_data={'flexflow': files},
  zip_safe= False,
  install_requires=['numpy>=1.16',
                    'cffi>=1.11',
                    'qualname>=0.1',
                    'keras_preprocessing',
                    'Pillow',
                    'cmake-build-extension',
                    'pybind11',
                    'ninja'
                    ],
  entry_points = {
          'console_scripts': ['flexflow_python=flexflow.driver:flexflow_driver'],
      },
  ext_modules=[
    CMakeExtension(name='flexflow',
                   install_prefix='flexflow',
                   cmake_configure_options=[
                       '-DFF_BUILD_FROM_PYPI=ON',
                       '-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF',
                       '-DFF_USE_PYTHON=ON',
                   ]),
  ],
  cmdclass=cmdclass,
  classifiers=[
      'Programming Language :: Python :: 3.6',
      'License :: OSI Approved :: Apache Software License',
      'Operating System :: POSIX :: Linux',
      'Topic :: Software Development :: Libraries',
  ],
  python_requires='>=3.6',
)
