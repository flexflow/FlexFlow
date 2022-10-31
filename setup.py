from setuptools import setup, find_packages
from pathlib import Path
from cmake_build_extension import BuildExtension, CMakeExtension
import os

datadir = Path(__file__).parent / "python/flexflow"
files = [str(p.relative_to(datadir)) for p in datadir.rglob("*.py")]

# Load CMake configs from config/config.linux file
configs_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "config.linux"
)


def parse_configs_from_file(configs_path):
    supported_cmake_vars = {
        "CC": "-DCMAKE_C_COMPILER",
        "CXX": "-DCMAKE_CXX_COMPILER",
        "INSTALL_DIR": "-DCMAKE_INSTALL_PREFIX",
        "BUILD_TYPE": "-DCMAKE_BUILD_TYPE",
        "FF_CUDA_ARCH": "-DFF_CUDA_ARCH",
        "CUDA_DIR": "-DCUDA_PATH",
        "CUDNN_DIR": "-DCUDNN_PATH",
        "FF_USE_NCCL": "-DFF_USE_NCCL",
        "FF_USE_GASNET": "-DFF_USE_GASNET",
        "FF_BUILD_ALL_EXAMPLES": "-DFF_BUILD_ALL_EXAMPLES",
        "FF_USE_AVX2": "-DFF_USE_AVX2",
        "FF_MAX_DIM": "-DFF_MAX_DIM",
    }
    envs_to_set = {}
    cmake_vars = {
        "-DFF_BUILD_FROM_PYPI": "ON",
        "-DCUDA_USE_STATIC_CUDA_RUNTIME": "OFF",
        "-DFF_USE_PYTHON": "ON",
        "-DFF_USE_NCCL": "ON",
        "-DFF_USE_GASNET": "ON",
        "-DFF_BUILD_ALL_EXAMPLES": "ON",
        "-DFF_USE_AVX2": "OFF",
    }
    gasnet_conduit = None
    with open(configs_path, "r") as f:
        for line in f.readlines():
            l = line.strip().split("=", 1)
            if len(l) == 2 and "#" not in l[0]:
                if len(l[1]) > 0:
                    if l[0] in supported_cmake_vars:
                        cmake_vars[supported_cmake_vars[l[0]]] = l[1]
                        # Special case
                        if l[0] == "CUDA_DIR":
                            envs_to_set["CUDA_PATH"] = os.path.join(
                                l[1], "lib64", "stubs"
                            )
                    elif l[0] == "FF_GASNET_CONDUIT":
                        gasnet_conduit = l[1]
    # Handle special cases
    if gasnet_conduit and cmake_vars["-DFF_USE_GASNET"] == "ON":
        cmake_vars["-DFF_GASNET_CONDUIT"] = gasnet_conduit
    cmake_vars = ["=".join((k, cmake_vars[k])) for k in cmake_vars]

    return envs_to_set, cmake_vars


envs_to_set, cmake_configure_options = parse_configs_from_file(configs_path)

# Export any relevant environment variables
for k in envs_to_set:
    os.environ[k] = envs_to_set[k]

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
    cmdclass={"build_ext": BuildExtension},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.6",
)
