# Running FlexFlow On Multiple Nodes
To build, install, and run FlexFlow on multiple nodes, follow the instructions below. We take AWS as an example to present the instructions.

## 1. Spin up instances
Spin up multiple instances with GPU support. We choose p3.2xlarge with [Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04)](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-neuron-pytorch-1-13-ubuntu-20-04/) to simplify the procedure.

Place the instances in a [placement group](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/placement-groups.html) which utilizes `cluster` as strategy to achieve the low-latency network performance.

To enable the communications between instances, you should attach the same security group to all instances and add an inbound rule in the security group to enable all the incoming traffic from the same security group. An example inbound rule is as follows:
```
Type: Custom TCP
Port range: 1 - 65535
Source: Custom (use the security group ID) 
```

## 2. Configure and build FlexFlow
Follow steps 1 to 5 in [INSTALL.md](INSTALL.md) to download the source code, install system dependencies, install the Python dependencies, configure the FlexFlow build, and build FlexFlow **on each instance**. You can skip the step 2 (Install system dependencies) if you have spun up instances with Deep Learning AMI which comes preconfigured with CUDA. Otherwise, you need to install system dependencies on each instance.

## 3. Test FlexFlow
Follow the step 6 in [INSTALL.md](INSTALL.md) to set environment variables.

A script to run a Python example on multiple nodes is available at `scripts/mnist_mlp_run.sh` and you can run the script using [`mpirun`](https://www.open-mpi.org/doc/current/man1/mpirun.1.php) or [`srun`](https://slurm.schedmd.com/srun.html). For example, to run the script with MPI, you need to first enable non-interactive `ssh` logins (refer to [Open MPI doc](https://docs.open-mpi.org/en/v5.0.0rc9/running-apps/ssh.html)) between instances and then run:
```
mpirun --host <host1_private_ip>:<slot1>,<host2_private_ip>:<slot2> -np <num_proc> ./scripts/mnist_mlp_run.sh
```

If you encounter some errors like `WARNING: Open MPI accepted a TCP connection from what appears to be a
another Open MPI process but cannot find a corresponding process
entry for that peer.`, add the parameter `--mca btl_tcp_if_include` in the `mpirun` command. (refer to [stack overflow question](https://stackoverflow.com/questions/15072563/running-mpi-on-two-hosts))