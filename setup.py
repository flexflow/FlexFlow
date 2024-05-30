from setuptools import setup, find_packages
from pathlib import Path
from cmake_build_extension import BuildExtension, CMakeExtension
import os, subprocess, requests, re
from datetime import date

datadir = Path(__file__).parent / "python/flexflow"
files = [str(p.relative_to(datadir)) for p in datadir.rglob("*.py")]

# Load CMake configs from config/config.linux file, parsing any custom settings from environment variables
configs_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "config.linux"
)
cmake_configure_options = subprocess.check_output([configs_path, "CMAKE_FLAGS"]).decode(
    "utf-8"
).strip().split() + ["-DFF_BUILD_FROM_PYPI=ON"]
cuda_path = subprocess.check_output([configs_path, "CUDA_PATH"]).decode("utf-8").strip()
# CUDA PATH should be passed to CMAKE via an environment variable
os.environ["CUDA_PATH"] = cuda_path

# set up make flags to parallelize build of subcomponents that do not use ninja
os.environ["MAKEFLAGS"] = (os.environ.get("MAKEFLAGS", "")) + f" -j{max(os.cpu_count()-1, 1)}" 

def compute_version() -> str:
    """This function generates the flexflow package version according to the following rules:
    1. If the python/flexflow/version.txt file exists, return the version from the file.
    2. If the version.txt file does not exist, the version will be YY.MM.<index>, 
        where the YY are the last two digits of the year, MM is the month number, 
        and <index> is a counter that is reset at the beginning of every month, 
        and it is incremented every time we publish a new version on pypi (or test.pypi, 
        if the DEPLOY_TO_TEST_PYPI env is defined and set to true). 
        Using this index (instead of the day of  the month) for the sub-subversion, allows 
        us to release more than once per day when needed.
    
    Warning! If the latest flexflow package version in test.pypi goes out of sync with pypi, this
    script will publish the wrong version if it is used to deploy to both test.pypi and pypi without
    deleting the version.txt file in-between the two uploads.

    :raises ValueError: if the python/flexflow/version.txt file exists, but contains a version in the wrong format
    :raises ValueError: if the DEPLOY_TO_TEST_PYPI env is set to a value that cannot be converted to a Python boolean
    :raises ValueError: if a flexflow release exists on pypi (or test.pypi) whose last two digits of the year are 
                        larger than the last two digits of the current year (e.g., if it's year '23, 
                        and we find a release from year '24)
    :return: The version in YY.MM.<incremental> format, as a string
    :rtype: str
    """    
    # Check if the version has already been determined before, in which case we don't recompute it
    version_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "python", "flexflow", "version.txt"
    )
    if os.path.isfile(version_file):
        with open(version_file) as f:
            version = f.read()
            # Version is YY.mm.<index>
            match = re.fullmatch(r'\d+\.\d+\.\d+', version)
            if not match:
                raise ValueError("Version is not in the right format!")
            return version

    # Get latest version of FlexFlow on pypi (default) or test.pypi (if the DEPLOY_TO_TEST_PYPI env is set to true)
    deploy_to_test_pypi = os.environ.get('DEPLOY_TO_TEST_PYPI', 'false')
    if deploy_to_test_pypi.lower() in ['true', 'yes', '1']:
        deploy_to_test_pypi = True
        pypi_url = "https://test.pypi.org/pypi/flexflow/json"
    elif deploy_to_test_pypi.lower() in ['false', 'no', '0']:
        deploy_to_test_pypi = False
        pypi_url = "https://pypi.org/pypi/flexflow/json"
    else:
        raise ValueError(f'Invalid boolean value: {deploy_to_test_pypi}')
    try:
        pip_version = requests.get(pypi_url).json()['info']['version']
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

# Create description from README
long_description = (Path(__file__).parent / "README.md").read_text()

# Create requirements list from requirements.txt
with open(Path(__file__).parent / "requirements.txt", "r") as reqs_file:
    requirements = reqs_file.read().strip().split("\n")

# Install Rust if not yet available
try:
    # Attempt to run a Rust command to check if Rust is installed
    subprocess.check_output(['cargo', '--version'])
except FileNotFoundError:
    print("Rust/Cargo not found, installing it...")
    # Rust is not installed, so install it using rustup
    try:
        subprocess.run("curl https://sh.rustup.rs -sSf | sh -s -- -y", shell=True, check=True)
        print("Rust and Cargo installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    # Add the cargo binary directory to the PATH
    os.environ["PATH"] = f"{os.path.join(os.environ.get('HOME', '/root'), '.cargo', 'bin')}:{os.environ.get('PATH', '')}"

setup(
    name="flexflow",
    version=compute_version(),
    description="A distributed deep learning framework that supports flexible parallelization strategies.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flexflow/FlexFlow",
    project_urls={
        "Homepage": "https://flexflow.ai/",
        "Documentation": "https://flexflow.readthedocs.io/en/latest/",
    },
    license="Apache",
    packages=find_packages("python"),
    package_dir={"": "python"},
    package_data={"flexflow": files},
    zip_safe=False,
    install_requires=requirements,
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
