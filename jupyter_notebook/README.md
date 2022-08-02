# Jupyter Notebook

This directory contains the Jupyter notebook support for
the FlexFlow Python binding.

## Quick Start
### Pre-requisite
* Python >= 3.6
* [Python binding of Legion](https://github.com/StanfordLegion/legion/tree/stable/bindings/python) 
  or [Legate](https://github.com/nv-legate/legate.core) needs to be installed.
* Install Jupyter notebook

        pip install notebook

### Install the Legion IPython kernel
```
python ./install.py --(configurations)
```
Please refer to the [Kernel Configurations](#kernel-configurations) section for the configuration details.

If the installtion is successed, the following log will be printed to the terminal.
The `flexflow_kernel_nocr` is the kernel name, and the `FlexFlow_SM_GPU` is the display name
of the kernel, which can be modified by the configuration json file. 
`FlexFlow` is the name entry in the json file, `SM` means the kernel
is only for shared memory machine, and `GPU` meams GPU execution is enabled. 
```
IPython kernel: flexflow_kernel_nocr(FlexFlow_SM_GPU) has been installed
```
The installed kernel can be also seen by using the following command:
```
jupyter kernelspec list
```

### Start the Jupyter Notebook server
* Launch jupyter notebook server on the computing node

        jupyter notebook --port=8888 --no-browser


* Create a turnel between localhost to the computing node, this step is not needed if
the jupyter server is run on the local machine.

        ssh -4 -t -L 8888:localhost:8002 username@login-node-hostname ssh -t -L 8002:localhost:8888 computing_node

### Use the Jupyter Notebook in the browser
* Open the browser, type the addredd http://localhost:8888/?token=xxx, the token will be
displayed in the terminal once the server is started. 
* Once the webpage is loaded, click "New" on the right top corner, and click the kernel 
just installed. It is shown as the display name of the kernel, e.g. `FlexFlow_SM_GPU`.

### Uninstall the kernel
```
jupyter kernelspec uninstall legion_kernel_nocr
```
If the kernel is re-installed, the old one will be automatically uninstalled by the install.py

## Kernel Configurations
The kernel can be configured by either passing arguments to `install.py` or using a json file.
The accepted arguments can be seen by 
```
python ./install.py --help
```

It is always preferred to use a json file. 
The `flexflow_python.json` is the template respect to the legate wrapper and
legion_python. Most entries are using the following format:
```
"cpus": {
    "cmd": "--cpus",
    "value": 1
}
```
* The `cpus` is the name of the field. 

* The `cmd` is used to tell how to pass the value to the field.
For example, flexflow uses `-ll:cpu` to set the number of CPUs, so the cmd in `flexflow_python.json` is `-ll:cpu`.

* The `value` is the value of the field. It can be set to `null`. In this case, the value is read
from the command line arguments. 

Other configuration options can be added by either appending them to the command line arguments or
using the `other_options` field of the json file. 
