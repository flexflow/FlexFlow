from setuptools import setup
from pathlib import Path

datadir = Path(__file__).parent / 'flexflow'
files = [str(p.relative_to(datadir)) for p in datadir.rglob('*.py')]

setup(
  name='flexflow',
  version='1.0',
  description='FlexFlow Python package',
  url='https://github.com/flexflow/FlexFlow',
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
      'Programming Language :: Python :: 3.6',
      'License :: OSI Approved :: Apache Software License',
      'Operating System :: POSIX :: Linux',
      'Topic :: Software Development :: Libraries',
  ],
  python_requires='>=3.6',
)
