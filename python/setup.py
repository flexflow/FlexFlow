from setuptools import setup
from pathlib import Path

datadir = Path(__file__).parent / 'flexflow'
files = [str(p.relative_to(datadir)) for p in datadir.rglob('*.py')]

setup(
  name='flexflow',
  version='1.0',
  description='FlexFlow Python package',
  url='https://github.com/flexflow/FlexFlow/tree/master/python/flexflow',
  license='Apache',
  packages=['flexflow'],
  package_data={'flexflow': files},
  install_requires=['numpy>=1.16',
                    'cffi>=1.11',
                    'qualname',
                    'keras_preprocessing',
                    'Pillow',
                    ],

  classifiers=[
      'License :: Apache License',
      'Operating System :: POSIX :: Linux',
      'Programming Language :: Python :: 3.6',
  ],
)