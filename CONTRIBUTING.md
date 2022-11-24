# Developers Guide

## Code Organization
The organization of FlexFlow repository is as follows (the core part of the code is stored in the bolded folders):

1. [.github](./.github): contains the Github CI workflows, and the pull request template
2. [align](./align): 
3. [cmake](./cmake): the CMake build helpers, included by [CMakeLists.txt](https://github.com/flexflow/FlexFlow/blob/master/CMakeLists.txt)
4. [conda](./conda): support for installing FlexFlow with `conda`
5. [config](./config): scripts used by CMake, `pip` and Docker to configure the FlexFlow builds
6. [deps](./deps): the submodule dependencies (GoogleTest, Json, Legion, NCCL, Optional, PyBind11 and Variant)
7. [docker](./docker): the docker containers and helper scripts
8. [docs](./docs): all the docs
9. **[examples](./examples): sample FlexFlow DNNs in C++ and Python**
10. [gdb](./gdb): 
11. **[include](./include): the FlexFlow include headers**
12. [jupyter_notebook](./jupyter_notebook): support for running FlexFlow in a Jupyter notebook
13. [nmt](./nmt): the original version of FlexFlow
14. [python](./python): the Python front-end interface
15. [scripts](./scripts): helper scripts for formatting and testing
16. [spack](./spack): helper file to use FlexFlow with the [Spack](https://spack.io/) package manager
17. **[src](./src): the FlexFlow source code**
18. [substitutions](./substitutions): 
19. [tests](./tests): the unit and integration tests
20. [tools](./tools): 

The best way to familiarize with the FlexFlow codebase is to walk through one of the existing examples, then check out the relevant FlexFlow runtime functions that are used in the example. 

## Continuous Integration
We currently implement CI testing using Github Workflows. Each workflow is defined by its corresponding YAML file in the [.github/workflows](.github/workflows) folder of the repo. We currently have the following workflows:

- `build.yml`: checks that the build & installation of FlexFlow succeed, using both the CMake and Makefile systems
- `clang-format-check.yml`: ensures that the source code is properly formatted.
- `docker-build.yml`: checks that the Docker containers can build and run FlexFlow properly
- `pip-install.yml`: checks the build & installation of FlexFlow using `pip`
- `shell-check.yml`: runs shellcheck on all bash scripts in the repo

We also have three placeholder workflows: `build-skip.yml`, `docker-build-skip.yml`, and `pip-install-skip.yml`. These always pass and are used only in the case of skipped workflows whose status is required to merge a PR; we implement the "hack" officially recommended by Github ([see here](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/troubleshooting-required-status-checks#handling-skipped-but-required-checks)).

### Github Workflow syntax
TODO: very quick summary of the main components of a Github workflow


## Contributing to FlexFlow
We want to make contributing to this project as easy and transparent as
possible.

### Formatting
We use `clang-format` to format our C++ code. If you make changes to the code and the Clang format CI test is failing, you can lint your code by running: `./scripts/format.sh` from the main folder of this repo.

### Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.

### Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

### License
By contributing to FlexFlow, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

