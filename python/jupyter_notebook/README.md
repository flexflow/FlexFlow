# Jupyter Notebook

This directory contains Jupyter notebook support for
FlexFlow. 
It allows user to run any FlexFlow Python
program (e.g., training models) on a single node using
the in-browser jupyter notebook UI. 

## Quick Start
### Pre-requisite
* Python >= 3.6
* FlexFlow Python binding needs to be installed, please check the [installation guide](https://github.com/flexflow/FlexFlow/blob/master/INSTALL.md)
* Install Jupyter notebook

        pip install notebook

### Install the FlexFlow IPython kernel
```
python ./install.py --(configurations)
```
Please refer to the [IPython Kernel Configurations](#kernel-configurations) section for the configuration details.

If the installation is successed, the following log will be printed to the terminal.
The `flexflow_kernel_nocr` is the IPython kernel name, where `nocr` means control replication is not enabled. 
The control replication can be enabled once multi-node jupyter notebook support is provided in the future. 
The `FlexFlow_SM_GPU` is the display name
of the kernel, which can be modified by the configuration json file. 
`FlexFlow` is the name entry in the json file, `SM` means the IPython kernel
is only for shared memory machine, and `GPU` means GPU execution is enabled. 
```
IPython kernel: flexflow_kernel_nocr(FlexFlow_SM_GPU) has been installed
```
The installed IPython kernel can be also seen by using the following command:
```
jupyter kernelspec list
```

### Create a turnel (Optional)
If you want to run the jupyter notebook server on a remote compute node instead of localhost, 
you can create a turnel from localhost to the compute node.
```
ssh -4 -t -L 8888:localhost:8002 username@login-node-hostname ssh -t -L 8002:localhost:8888 computing_node
```

### Start the Jupyter Notebook server
Launch jupyter notebook server on the compute node or localhost if the turnel is not created
```
jupyter notebook --port=8888 --no-browser
```

### Use the Jupyter Notebook in the browser
* Open the browser, type the addredd http://localhost:8888/?token=xxx, the token will be
displayed in the terminal once the server is started. 
* Once the webpage is loaded, click "New" on the right top corner, and click the kernel 
just installed. It is shown as the display name of the kernel, e.g. `FlexFlow_SM_GPU`.

### Uninstall the IPython kernel
```
jupyter kernelspec uninstall flexflow_kernel_nocr
```
If the IPython kernel is re-installed, the old one will be automatically uninstalled by the install.py


## IPython Kernel Configurations
The IPython kernel can be configured by either passing arguments to `install.py` or using a json file.
The accepted arguments can be listed with
```
python ./install.py --help
```

It is always preferred to use a json file. 
The `flexflow_python.json` is the template respect to the
flexflow_python. Most entries are using the following format:
```
"cpus": {
    "cmd": "--cpus",
    "value": 1
}
```
* `cpus` is the name of the field. 

* `cmd` is used to tell how to pass the value to the field.
For example, flexflow uses `-ll:cpu` to set the number of CPUs, so the `cmd` in `flexflow_python.json` is `-ll:cpu`.

* `value` is the value of the field. It can be set to `null`. In this case, the value is read
from the command line arguments. 

Other configuration options can be added by either appending them to the command line arguments or
using the `other_options` field of the json file. 
