from setuptools import setup
from pathlib import Path
from cmake_build_extension import BuildExtension, CMakeExtension

datadir = Path(__file__).parent / 'python/flexflow'
files = [str(p.relative_to(datadir)) for p in datadir.rglob('*.py')]

setup(
  name='flexflow',
  version='1.0',
  description='FlexFlow Python package',
  url='https://github.com/flexflow/FlexFlow',
  license='Apache',
  packages=['flexflow'],
  package_data={'flexflow': files},
  zip_safe= False,
  install_requires=['numpy>=1.16',
                    'cffi>=1.11',
                    'qualname',
                    'keras_preprocessing',
                    'Pillow',
                    'cmake-build-extension',
                    'pybind11'
                    ],
  ext_modules=[
    CMakeExtension(name='flexflow',
                   install_prefix='flexflow',
                   cmake_configure_options=[
                       '-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF -DCUDA_PATH=/projects/opt/centos7/cuda/10.1 -DCUDNN_PATH=/projects/opt/centos7/cuda/10.1 -DFF_USE_PYTHON=ON -DFF_USE_NCCL=OFF -DFF_USE_GASNET=OFF -DFF_BUILD_EXAMPLES=OFF -DFF_USE_AVX2=OFF -DFF_MAX_DIM=4',
                   ]),
  ],
  cmdclass=dict(build_ext=BuildExtension),
  classifiers=[
      'Programming Language :: Python :: 3.6',
      'License :: OSI Approved :: Apache Software License',
      'Operating System :: POSIX :: Linux',
      'Topic :: Software Development :: Libraries',
  ],
  python_requires='>=3.6',
)
