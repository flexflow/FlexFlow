# Developers Guide

## Code Organization
The bulk of the FlexFlow source code is stored in the following folders:

1. `examples`: example DNNs in C++ and Python
2. `include`: the FlexFlow headers
3. `src`: the FlexFlow source code
4. `python`: bindings for the Python interface

The `src` folder is divided into the following subfolders:

* `loss_functions`: contains the implementation of all the supported loss functions, as well as the backward function to be used during training.
* `mapper`: contains the implentation of the Legion custom mapper for FlexFlow, `FFMapper`.
* `metric_functions`: contains the implementation of all the metrics functions, such as accuracy, categorical crossentropy, or mean squared error.
* `ops`: contains the implementation of all tensor operators.
* `parallel_ops`: contains the operators used to represent parallelization on the Parallel Computation Graph (PCG) as described in the [Unity paper](https://www.usenix.org/system/files/osdi22-unger.pdf).
* `recompile`: support for the dynamic recompilation functionality described in [this paper](https://arxiv.org/pdf/2205.01848.pdf)
* `runtime`: contains the implementation of the high-level FlexFlow runtime
* `utils`: only contains implementation of the RecordFormatter class.

In many parts of the source code you will see triplets of files with the following three different extensions: `.cc`, `.cpp` and `.cu`. The `.cc` file contains the main, high-level C++ implementation, whereas the `.cpp` and `.cu` file contain, respectively, the HIP and CUDA kernels.

The best way to familiarize with the FlexFlow codebase is to walk through one of the existing examples, then check out the relevant FlexFlow runtime functions that are used in the example.

### AlexNet example (C++)

[TODO]: In this section, we will walk through the AlexNet C++ implementation, which can be found in the [examples/cpp/AlexNet](https://github.com/flexflow/FlexFlow/tree/master/examples/cpp/AlexNet) folder of the repository.


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

