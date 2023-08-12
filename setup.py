from setuptools import setup, find_packages
from pathlib import Path
from cmake_build_extension import BuildExtension, CMakeExtension
import os, subprocess, requests, re
from datetime import date

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
# CUDA PATH should be passed to CMAKE via an environment variable
os.environ["CUDA_PATH"] = cuda_path

# set up make flags
os.environ["MAKEFLAGS"] = (os.environ.get("MAKEFLAGS", "")) + f" -j{max(os.cpu_count()-1, 1)}" 

def compute_version():
    # Check if the version has already been determined before, in which case we don't recompute it
    version_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "python", "flexflow", "version.txt"
    )
    if os.path.isfile(version_file):
        with open(version_file) as f:
            version = f.read()
            match = re.fullmatch(r'\d+\.\d+\.\d+', version)
            if not match:
                raise ValueError("Version is not in the right format!")
            return version

    # Version is YY.mm.<incremental>
    # TODO: replace testpypi repo with pypi repo
    # pip_version = requests.get("https://pypi.org/pypi/flexflow/json").json()['info']['version']
    try:
        pip_version = requests.get("https://test.pypi.org/pypi/flexflow/json").json()['info']['version']
    except KeyError:
        pip_version = "0.0.0"

    pip_year, pip_month, pip_incremental = [int(x) for x in pip_version.split(".")]

    today = date.today()
    year_two_digits = int(str(today.year)[-2:])
    
    # Ensure no version from the distant past or the future :)
    if pip_year > year_two_digits or (pip_year == year_two_digits and pip_month > today.month):
        raise ValueError(f"A version from the distant past or future (year '{pip_year}, month {pip_month}) already exists!")
    
    subversion = 0
    if pip_year == year_two_digits and pip_month == today.month:
        subversion = pip_incremental + 1

    version = f"{year_two_digits}.{today.month}.{subversion}"
    # Add version to file
    with open(version_file, 'w+') as f:
        f.write(version)

    return version

setup(
    name="flexflow",
    version=compute_version(),
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
        "requests",
    ],
    scripts=['python/flexflow/flexflow_python'],
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
)
