# Running FlexFlow on Multiple Nodes

To build, install, and run FlexFlow on multiple nodes, follow the instructions below. We will use AWS as an example to present the instructions.

## 1. Spin up instances

Spin up multiple instances with GPU support. For AWS, we recommend using p3.2xlarge with [Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04)](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-neuron-pytorch-1-13-ubuntu-20-04/) to simplify the procedure.

Place the instances in a [placement group](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/placement-groups.html) that utilizes the `cluster` strategy to achieve low-latency network performance.

To enable communication between instances, attach the same security group to all instances and add an inbound rule in the security group to allow all incoming traffic from the same security group. An example inbound rule is as follows:
```
Type: Custom TCP
Port range: 1 - 65535
Source: Custom (use the security group ID) 
```

You can also use your own GPU cluster, as long as all machines are interconnected with a low-latency network.

## 2. Configure and build FlexFlow

Follow steps 1 to 5 in the [Build from source guide](https://flexflow.readthedocs.io/en/latest/installation.html) to download the source code, install system dependencies, install the Python dependencies, configure the FlexFlow build, and build FlexFlow **on each instance at the same path**. 

You can skip step 2 (Install system dependencies) if you have spun up instances with Deep Learning AMI, which comes preconfigured with CUDA. Otherwise, you need to install system dependencies on each instance.

For step 4 (Configuring the FlexFlow build), make sure to specify a network using the `FF_LEGION_NETWORKS` parameter. We recommend using `FF_LEGION_NETWORKS=gasnet` and `FF_GASNET_CONDUIT=ucx`. Other configurations are optional.

## 3. Configure MPI

MPI is an easy way to launch FlexFlow across all instances simultaneously and set up communication between them.

To use MPI, enable non-interactive `ssh` logins between instances. This can be done by referring to the [Open MPI documentation](https://docs.open-mpi.org/en/v5.0.0rc9/running-apps/ssh.html). Here are the detailed steps:

1. Choose one of the nodes as the main instance and create a public/private key pair on the instance. This will be the instance from which you launch MPI commands. Run the following command:

```
ssh-keygen -t ed25519
```

This will create a public key at `~/.ssh/id_ed25519.pub` and a private key at `~/.ssh/id_ed25519`.

2. Append the contents of the **public key** to `~/.ssh/authorized_keys` on all machines (if the file does not exist, create one). Execute the following command on **all instances**:

```
mkdir -p ~/.ssh
echo '<public key>' >> ~/.ssh/authorized_keys
```

Replace `<public key>` with the public key from `~/.ssh/id_ed25519.pub` on the main instance. It should be a single line containing a string like:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOy5NKYdE8Cwgid59rx6xMqyj9vLaWuXIwy/BSRiK4su instance
```

3. Create a hostfile at `~/hostfile`, with one line for each instance (add more lines if you have more instances):

```
<host1_private_ip> slots=<slot1>
<host2_private_ip> slots=<slot2>
```

`<slot1>` and `<slot2>` refer to the number of slots available for each instance, respectively. Set it to one if you have a GPU on each instance.

4. SSH into each host and make sure you can log into them. It may ask you to verify the public key. Make sure to trust the public key so that it doesn't ask you again.

5. Test MPI by running `mpirun -N 1 --hostfile ~/hostfile hostname`. It should display the hostname of all your nodes. If you encounter any errors like `WARNING: Open MPI accepted a TCP connection from what appears to be another Open MPI process but cannot find a corresponding process entry for that peer.`, add the parameter `--mca btl_tcp_if_include` in the `mpirun` command (refer to [this Stack Overflow question](https://stackoverflow.com/questions/15072563/running-mpi-on-two-hosts)).

## 4. Test FlexFlow

Follow step 6 in the [Build from source guide](https://flexflow.readthedocs.io/en/latest/installation.html) to set environment variables.

A script to run a Python example on multiple nodes is available at `scripts/mnist_mlp_run.sh`. You can run the script using [`mpirun`](https://www.open-mpi.org/doc/current/man1/mpirun.1.php) (if you configured it in step 3) or [`srun`](https://slurm.schedmd.com/srun.html).
