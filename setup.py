from setuptools import setup, find_packages
from pathlib import Path
from cmake_build_extension import BuildExtension, CMakeExtension
import os
import subprocess

datadir = Path(__file__).parent / "python/flexflow"
files = [str(p.relative_to(datadir)) for p in datadir.rglob("*.py")]

# Load CMake configs from config/config.linux file
configs_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "config.linux"
)

cmake_configure_options = subprocess.check_output([configs_path, "CMAKE_FLAGS"]).decode(
    "utf-8"
).strip().split() + ["-DFF_BUILD_FROM_PYPI=ON"]
cuda_path = subprocess.check_output([configs_path, "CUDA_PATH"]).decode("utf-8").strip()

os.environ["CUDA_PATH"] = cuda_path

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
