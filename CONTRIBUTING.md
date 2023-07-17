# Developers Guide

## Code Organization
The bulk of the FlexFlow source code is stored in the following folders:

1. `examples`: example DNNs in C++ and Python
2. `include`: the FlexFlow headers
3. `src`: the FlexFlow source code
4. `python`: bindings for the Python interface

The `src` folder is divided into the following subfolders:

1. `loss_functions`: contains the implementation of all the supported loss functions, as well as the backward function to be used during training.
2. `mapper`: contains the implentation of the Legion custom mapper for FlexFlow, `FFMapper`.
3. `metric_functions`: contains the implementation of all the metrics functions, such as accuracy, categorical crossentropy, or mean squared error.
4. `ops`: contains the implementation of all tensor operators.
5. `parallel_ops`: contains the operators used to represent parallelization on the Parallel Computation Graph (PCG) as described in the [Unity paper](https://www.usenix.org/system/files/osdi22-unger.pdf).
6. `recompile`: support for the dynamic recompilation functionality described in [this paper](https://arxiv.org/pdf/2205.01848.pdf)
7. `runtime`: contains the implementation of the high-level FlexFlow runtime
8. `utils`: only contains implementation of the RecordFormatter class.

In many parts of the source code you will see triplets of files with the following three different extensions: `.cc`, `.cpp` and `.cu`. The `.cc` file contains the main, high-level C++ implementation, whereas the `.cpp` and `.cu` file contain, respectively, the HIP and CUDA kernels.

The best way to familiarize with the FlexFlow codebase is to walk through one of the existing examples, then check out the relevant FlexFlow runtime functions that are used in the example. We provide examples in both Python and C++. The Python interface is the most up-to-date, and the one that is intended to be used by users. To learn how to _run_ a DNN in FlexFlow, please refer to the scripts in the [examples/python](https://github.com/flexflow/FlexFlow/tree/master/examples/python) folder. The C++ interface is intended mostly for development purposes and may have some rough edges. Nevertheless, the C++ examples are the preferred ones to look at if you want to familiarize with the internals of the FlexFlow implementation. 

### AlexNet example (C++)

In this section, we will walk through the AlexNet C++ implementation, which can be found in the [examples/cpp/AlexNet](https://github.com/flexflow/FlexFlow/tree/master/examples/cpp/AlexNet) folder of the repository. You can use this example as a template to write your own C++ DNN model using FlexFlow. 

You can start by taking a look at the `alexnet.cc` file, containing the core of the implementation. You will notice the absence of a `main()` function. The FlexFlow C++ interface uses the `main()` function defined in [src/runtime/cpp_driver.cc](https://github.com/flexflow/FlexFlow/blob/master/src/runtime/cpp_driver.cc), so you will not need to create a new one when writing a FlexFlow program. Instead, you will use a function called `top_level_task` and with the following signature:

```c++
void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime);
```

Inside the `top_level_task` function, you will want to create a FFModel object, which is usually initialized by passing a FFConfig object to the constructor:

```c++
FFConfig ffConfig;
FFModel ff(ffConfig);
```

`FFModel` is a very large class, and is the cornerstone of every FlexFlow DNN, providing the methods required to instantiate input tensors, add layers, compile the model, etc... 

#### Tensor creation

The typical first step in a FlexFlow DNN is to define the input tensors. You can do that using the `FFModel.create_tensor` function. In the case of AlexNet:

```c++
Tensor input;
{
	int const dims[] = {ffConfig.batchSize, 3, 229, 229};
	input = ff.create_tensor<4>(dims, DT_FLOAT);
}
```  

In the case of AlexNet, the input tensor has dimension `batch_size x 3 x 229 x 229`, so it is a 4-dimensional tensor. To initialize the tensor, we use the templated `create_tensor` function, which is part of `FFModel`. It may be useful to know that the `create_tensor` function lays out the tensor's dimensions in reverse order. For instance, in the snippet above, printing the `input` tensor (which can be done using the instruction below) will show dimensions: `[229, 229, 3, batch_size]`. 

```c++
input->print("input tensor")
``` 

There are two versions of the `create_tensor` function: one (used in the last snippet above) uses a template that takes the number of tensor dimensions as its parameter; the second is a wrapper around the first, and takes the number of tensor dimensions as a regular function parameter. Both versions are implemented in `model.cc`, and their signature is identical, except for the number of dimensions parameter. Below, we discuss the implementation of the `create_tensor` wrapper, since it illustrates a common pattern among FlexFlow functions:

```c++
Tensor FFModel::create_tensor(int numdim,
                              int const dims[],
                              DataType data_type,
                              Layer const *layer,
                              int idx,
                              bool create_grad) {
  switch (numdim) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    return create_tensor<DIM>(dims, data_type, layer, idx, create_grad);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported dim!");
  }
}
```

The `LEGION_FOREACH_N(DIMFUNC)` macro is defined in [deps/legion/runtime/legion/legion_config.h](https://gitlab.com/StanfordLegion/legion/-/blob/master/runtime/legion/legion_config.h). The preprocessor replaces the block of code between `#define DIMFUNC(DIM)` and `#undef DIMFUNC` with a `case` statement for each integer between `1` and the `LEGION_MAX_DIM`, controlled by the `Legion_MAX_DIM` Legion CMake variable, which in case of FlexFlow, is set equal to `FF_MAX_DIM` in [cmake/legion.cmake](https://github.com/flexflow/FlexFlow/blob/master/cmake/legion.cmake). For example, in the default case, where `FF_MAX_DIM` is set to 4, the preprocessor will rewrite the `switch` loop above as follows:

```c++
switch (numdim) {
	case 1:
		return create_tensor<1>(dims, data_type, layer, idx, create_grad);
	case 2:
		return create_tensor<2>(dims, data_type, layer, idx, create_grad);
	case 3:
		return create_tensor<3>(dims, data_type, layer, idx, create_grad);
	case 4:
		return create_tensor<4>(dims, data_type, layer, idx, create_grad);
	default:
		assert(false && "Unsupported dim!");
}
```

In addition to the two versions of `create_tensor` discussed above, `model.cc` also offers the `create_tensor_legion_ordering` function, which simply creates a tensor without reversing the order of the input dimensions. The explicit template instantiations at the bottom of `model.cc` will ensure that functions such `create_tensor` are only instantiated for number of dimensions that are less or equal to `FF_MAX_DIM`.

#### Adding layers to a DNN model

Going back to the AlexNet example, after defining the input tensors, we can add each of the DNN's layers by using the corresponding method from `FFModel`. For instance, the first layer is added using: 

```c++
t = ff.conv2d(input, 64, 11, 11, 4, 4, 2, 2, AC_MODE_RELU);
```
The `conv2d` function is defined in [src/ops/conv_2d.cc](https://github.com/flexflow/FlexFlow/blob/master/src/ops/conv_2d.cc). Just like the other `FFModel` layer functions, it creates a new `Layer` object, populates with all relevant properties, and then enqueues to the list of layers in the `FFModel` class. 

#### Optimizer and training metrics

After adding the DNN layers, the next step before compiling the model for training is to initialize an optimizer and then create a vector with all the metrics that you want to monitor at each training step.


#### Model compilation

TODO


## Continuous Integration
We currently implement CI testing using Github Workflows. Each workflow is defined by its corresponding YAML file in the [.github/workflows](.github/workflows) folder of the repo. We currently have the following workflows:

1. `build.yml`: checks that the build & installation of FlexFlow succeed, using both the CMake and Makefile systems
2. `clang-format-check.yml`: ensures that the source code is properly formatted.
3. `docker-build.yml`: checks that the Docker containers can build and run FlexFlow properly. It also publishes a new version of the FlexFlow containers to the repo's package register for each push to the master branch
4. `gpu-ci.yml`: runs all the tests that require a GPU to run.
5. `gpu-ci-daemon.yml`: an helper workflow that turns on/off the GPU instance used by the test above
6. `multinode-test.yml`: runs the same GPU tests from the `gpu-ci.yml` workflow, but using multiple (simulated) nodes. The test currently simulates two nodes, each with 2 GPUs. To run FlexFlow on multiple nodes, we compile Legion with GASNET enabled, and choose MPI as the GASNET conduit. Compared to the single-node version, this test is much more time-consuming (about 4h instead 40mins at the time of writing), so we only run the test on the FlexFlow `master` branch every other day.
7. `pip-install.yml`: checks the build & installation of FlexFlow using `pip`
8. `shell-check.yml`: runs shellcheck on all bash scripts in the repo

We also have three placeholder workflows: `build-skip.yml`, `docker-build-skip.yml`, `gpu-ci-skip` and `pip-install-skip.yml`. These always pass and are used only in the case of skipped workflows whose status is required to merge a PR; we implement the "hack" officially recommended by Github ([see here](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/troubleshooting-required-status-checks#handling-skipped-but-required-checks)).

In the next section, we walk through an example workflow, similar to the ones found in this repo. An important thing to note is that Github workflows do not run unless they are properly linted. If you encounter a formatting/linting error, you can lint your workflow file using `prettier` (installation instructions [here](https://prettier.io/docs/en/install.html)):

```bash
yarn prettier --write <filename.yml>
```

### Github Workflow syntax

In this section, we will walk through an example workflow:

```yaml
name: "build"

on:
  pull_request:
    paths:
      - "src/**"
      - ".github/workflows/build.yml"
  push:
    paths:
      - "src/**"
      - ".github/workflows/build.yml"
    branches:
      - "master"
  schedule:
    # Run weekly on Saturday at midnight PT (3am ET / 8am UTC)
    - cron: "0 8 * * 6"
  workflow_dispatch:

concurrency:
  group: build-${{ github.head_ref || github.run_id }}
  cancel-in-progress: true

jobs:
  cmake-build:
    name: Build FlexFlow with CMake
    runs-on: ubuntu-20.04
    steps:
      - name: Checkout Git Repository
        uses: actions/checkout@v3
        with:
          submodules: recursive

      - name: Install CUDA
        uses: Jimver/cuda-toolkit@v0.2.11
        id: cuda-toolkit
        with:
          cuda: "11.8.0"
          # Disable caching of the CUDA binaries, since it does not give us any significant performance improvement
          use-github-cache: "false"

      - name: Install FlexFlow Dependencies
        run: .github/workflows/helpers/install_dependencies.sh
```

The first instruction in a workflow file sets the workflow's name. The name is not required to be unique, but it is preferrable to use unique names to avoid conflicts. 

Next, the `on:` section allows you to control what events trigger a workflow run. A full list of events that can trigger a workflow run is available [here](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows). Each trigger can take options that further filter out the scenarios where the workflow runs. In the example above, we have the following triggers: 

1. A `pull_request` trigger, triggering a workflow run when a PR is opened, and for each new commit to a branch associated with an open PR. The `paths` option allows you to choose which files in the repository need to be modified to make the workflow run. For instance, in the example, the `pull_request` trigger is only activated for PRs where either `.github/workflows/build.yml` or a file in the `src` folder is modified. 
2. A `push` trigger, triggering a run for each push, no matter if there is an open PR or not. Here, in addition to the `paths` option, we have a `branches` option, restricting the trigger to activate only for commits to the `master` branch, but not for commits to other branches.
3. A `schedule` trigger, triggering the workflow at specific times. The syntax for chron workflows is explained [here](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#onschedule).
4. A `workflow_dispatch` trigger, enabling authorized users to manually run the workflow.

There are many additional options that are not discussed here. For example, there is a `paths-ignore` option that allows you to run the workflow in any case except if a file at the specified paths is modified.

Next, the `concurrency` section allows you to control how many copies of the same workflow can run in parallel. This is useful, for example, when one pushes a new commit to a branch before the workflows for the previous commits have finished running. Since the old commit is now obsolete, there is no need to wait until the old workflow has finished running before running again on the newer commit. In the example above, for example, we use the `concurrency` section to cancel any queued or in-progress workflow when a newer one is triggered.

Finally, we define the jobs that will run when the workflow is triggered. Each job is specified by adding an indented entry to the `jobs:` section, and will run in parallel in a isolated container. Multiple jobs in the same workflow do not directly share files. The `runs-on` option allows you to control what type of runner to use for the job. In the example, we use `runs-on: ubuntu-20.04` to run the job on a VM with Ubuntu 20.04. You can also set up the workflow to run on a self-hosted machine by using the option `runs-on: self-hosted` and following the instructions at [this link](https://docs.github.com/en/actions/hosting-your-own-runners/adding-self-hosted-runners) to connect the self hosted machine to the repository. 

Each step in a job will be executed sequentially, and if it fails, the remaining steps will be cancelled and the job will be marked as `failed`. Each step is specified by either reusing a Github action or running a shell command (or a script file). For instance, in the example above, the first step uses the Github Action `actions/checkout@v3` to check out the repository, the second step uses the `Jimver/cuda-toolkit@v0.2.11` action to install CUDA, whereas the third step runs a bash script stored in the repo at the path `.github/workflows/helpers/install_dependencies.sh`.

## Contributing to FlexFlow
We want to make contributing to this project as easy and transparent as possible.

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

