from setuptools import setup, find_packages
from pathlib import Path
from cmake_build_extension import BuildExtension, CMakeExtension
import subprocess, sys, os

datadir = Path(__file__).parent / "python/flexflow"
files = [str(p.relative_to(datadir)) for p in datadir.rglob("*.py")]

# Load CMake configs from config/config.linux file
configs_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "config.linux"
)
output = subprocess.check_output(
    f". {configs_path} && ( set -o posix ; set ) | cat", shell=True
)
cfg = dict(
    [
        line.split("=", 1)
        for line in output.decode().splitlines()
        if line and len((line or "").split("=", 1)) == 2
    ]
)
# Pip supports the configs in the two lists below
support_env_vars = [
    "CC_FLAGS",
    "NVCC_FLAGS",
    "LD_FLAGS",
    "CUDA_PATH",
    "CUDA_DIR",
    "CUDNN_DIR",
]
supported_cmake_vars = [
    "SET_CC",
    "SET_CXX",
    "SET_INSTALL_DIR",
    "SET_BUILD",
    "SET_CUDA_ARCH",
    "SET_CUDA",
    "SET_CUDNN",
    "SET_PYTHON",
    "SET_PIP",
    "SET_PYBIND11",
    "SET_NCCL",
    "SET_GASNET",
    "SET_EXAMPLES",
    "SET_AVX2",
    "SET_MAX_DIM",
]
# If the config file sets any variable whose name appears in the support_env_vars list,
# export such variable before running CMake
for k in support_env_vars:
    if k in cfg:
        os.environ[k] = cfg[k]
# If the config file sets any variable whose name appears in the supported_cmake_vars list,
# pass such variable as a config option to CMake, in addition to the two default configs
# '-DFF_BUILD_FROM_PYPI=ON' and '-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF'.
cmake_configure_options = [
    "-DFF_BUILD_FROM_PYPI=ON",
    "-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF",
] + [cfg[k] for k in supported_cmake_vars if k in cfg]

setup(
    name="flexflow",
    version="1.0",
    description="FlexFlow Python package",
    url="https://github.com/flexflow/FlexFlow",
    license="Apache",
    packages=find_packages("python"),
    package_dir={"": "python"},
    package_data={"flexflow": files},
    zip_safe=False,
    install_requires=[
        "numpy>=1.16",
        "cffi>=1.11",
        "qualname>=0.1",
        "keras_preprocessing",
        "Pillow",
        "cmake-build-extension",
        "pybind11",
        "ninja",
    ],
    entry_points={
        "console_scripts": ["flexflow_python=flexflow.driver:flexflow_driver"],
    },
    ext_modules=[
        CMakeExtension(
            name="flexflow",
            install_prefix="flexflow",
            cmake_configure_options=cmake_configure_options,
        ),
    ],
    cmdclass=dict({"build_ext": BuildExtension}),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.6",
)
