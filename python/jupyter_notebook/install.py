#!/usr/bin/env python3

# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import os
import re
import sys
import argparse
from distutils import log
import json
import inspect
import shutil

from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel
from IPython.utils.tempdir import TemporaryDirectory

kernel_json = {"argv": [],
    "display_name": "None",
    "language": "python",
}

kernel_json_suffix_nocr = ["flexflow_kernel_nocr.py", "-f", "{connection_file}"]


required_cmd_dict_key = ["name", "kernel_name", "flexflow_python_prefix", "exe", "cpus", "gpus", "openmp", "ompthreads", "utility", "sysmem", "fbmem", "zcmem", "regmem", "not_control_replicable"]

# This internal method is used to delete a kernel specified by kernel_name
def _delete_kernel(ksm, kernel_name, mute=True):
    try:
        spec = ksm.get_kernel_spec(kernel_name)
        shutil.rmtree(spec.resource_dir)
        if mute == False:
            print("Find existing kernel:" + kernel_name + ", delete it before installation.")
    except NoSuchKernel:
        if mute == False:
            print("No existing kernel:" + kernel_name + " has been installed, continue to installation.")

# This internal method is used to install a kernel
def _install_kernel(ksm, kernel_name, kernel_json, user, prefix, mute=True):
    with TemporaryDirectory() as td:
        os.chmod(td, 0o755)
        with open(os.path.join(td, "kernel.json"), "w") as f:
            json.dump(kernel_json, f, sort_keys=True)
        try:
            ksm.install_kernel_spec(td, kernel_name, user=user, prefix=prefix)
            if mute == False:
                print("IPython kernel: " + kernel_name + "(" + kernel_json["display_name"] + ") has been installed")
        except Exception as e:
            if mute == False:
                log.error("Failed to install the IPython kernel: " +  kernel_name + "(" + kernel_json["display_name"] + ") with error: " + str(e))

# This method parses the json file into a dict named cmd_dict
def parse_json(flexflow_python_prefix,
               cpus, 
               gpus,
               openmp,
               ompthreads,
               utility,
               sysmem,
               fbmem,
               zcmem,
               regmem,
               launcher,
               nodes,
               ranks_per_node,
               not_control_replicable,
               kernel_name,
               filename):
    with open(filename) as json_file:
        cmd_dict = json.load(json_file)
        for key in required_cmd_dict_key:
            if key not in cmd_dict:
                assert 0, "Key: " + key + " is not existed."
    # Criterion
    #   if entry in the json file is set to null, we load it from the cmd line
    args = inspect.getfullargspec(parse_json)
    keys = args.args[0: len(args.args)-1]
    sig = inspect.signature(parse_json)
    argv_dict = locals()
    for key in keys:
        if key == "launcher":
            if cmd_dict[key]["value"] == None and argv_dict[key] != "none":
                cmd_dict[key]["value"] = argv_dict[key]
            if cmd_dict[key]["launcher_extra"] == None:
                cmd_dict[key]["launcher_extra"] = list()
        elif key == "flexflow_python_prefix" or key == "kernel_name":
            if cmd_dict[key] == None:
                cmd_dict[key] = argv_dict[key]
        else:
            if cmd_dict[key]["value"] == None:
                cmd_dict[key]["value"] = argv_dict[key]

    return cmd_dict

# This method is used to install the kernel for jupyter notebook support for single or
# multiple nodes runs without control replication
def install_kernel_nocr(user, prefix, cmd_opts, cmd_dict, verbose, kernel_file_dir):
    if verbose:
        print("cmd_dict is:\n" + str(cmd_dict))

    # setup name and argv
    kernel_json["argv"] = [cmd_dict["flexflow_python_prefix"] + "/" + cmd_dict["exe"]] + kernel_json["argv"]
    kernel_json["display_name"] = cmd_dict["name"]

    # launcher
    if cmd_dict["launcher"]["value"] == None:
        kernel_json["display_name"] += "_SM"
    else:
        kernel_json["display_name"] += "_DM"
        nodes = cmd_dict["nodes"]["value"]
        ranks_per_node = cmd_dict["ranks_per_node"]["value"]
        launcher = cmd_dict["launcher"]["value"]
        if cmd_dict["launcher"]["type"] == "legate":
            # use legate launcher
            kernel_json["argv"] += cmd_dict["launcher"]["cmd"], launcher, \
                                   cmd_dict["nodes"]["cmd"], str(nodes), \
                                   cmd_dict["ranks_per_node"]["cmd"], str(ranks_per_node)
        else:
            # use mpirun, srun and jsrun launcher
            ranks = nodes * ranks_per_node
            if launcher == "mpirun":
                kernel_json["argv"] = ["mpirun", "-n", str(ranks), "--npernode", str(ranks_per_node)] + cmd_dict["launcher"]["launcher_extra"] + kernel_json["argv"]
            elif launcher == "srun":
                kernel_json["argv"] = ["srun", "-n", str(ranks), "--ntasks-per-node", str(ranks_per_node)] + cmd_dict["launcher"]["launcher_extra"] + kernel_json["argv"]
            elif launcher == "jsrun":
                kernel_json["argv"] = ["jsrun", "-n", str(ranks // ranks_per_node), "-r", "1", "-a", str(ranks_per_node)] + cmd_dict["launcher"]["launcher_extra"] + kernel_json["argv"]
            else:
                assert 0, "Unknown launcher"

    # let's do not enable control replication because pygion has issue with cleaning up
    # disable control replication
    # assert cmd_dict["not_control_replicable"]["value"] == True
    # kernel_json["argv"].append(cmd_dict["not_control_replicable"]["cmd"])

    # cpu
    if cmd_dict["cpus"]["value"] > 0:
        kernel_json["argv"] += cmd_dict["cpus"]["cmd"], str(cmd_dict["cpus"]["value"])

    # gpu
    if cmd_dict["gpus"]["value"] > 0:
        kernel_json["display_name"] += "_GPU"
        kernel_json["argv"] += cmd_dict["gpus"]["cmd"], str(cmd_dict["gpus"]["value"])
        if cmd_dict["fbmem"]["value"] > 0:
            kernel_json["argv"] += cmd_dict["fbmem"]["cmd"], str(cmd_dict["fbmem"]["value"])
        if cmd_dict["zcmem"]["value"] > 0:
            kernel_json["argv"] += cmd_dict["zcmem"]["cmd"], str(cmd_dict["zcmem"]["value"])

    # openmp
    if cmd_dict["openmp"]["value"] > 0:
        if cmd_dict["ompthreads"]["value"] > 0:
            kernel_json["argv"] += cmd_dict["openmp"]["cmd"], str(cmd_dict["openmp"]["value"])
            kernel_json["argv"] += cmd_dict["ompthreads"]["cmd"], str(cmd_dict["ompthreads"]["value"])
        else:
            print(
                "WARNING: ignore request for "
                + str(cmd_dict["openmp"]["value"])
                + "OpenMP processors with 0 threads"
            )
    
    # utility
    if cmd_dict["utility"]["value"] > 0:
        kernel_json["argv"] += cmd_dict["utility"]["cmd"], str(cmd_dict["utility"]["value"])
    
    # system memory
    if cmd_dict["sysmem"]["value"] > 0:
        kernel_json["argv"] += cmd_dict["sysmem"]["cmd"], str(cmd_dict["sysmem"]["value"])
    
    # register memory
    if cmd_dict["regmem"]["value"] > 0:
        kernel_json["argv"] += cmd_dict["regmem"]["cmd"], str(cmd_dict["regmem"]["value"])
    
    # other options from json
    if "other_options" in cmd_dict:
        other_options = cmd_dict["other_options"]
        for option in other_options:
            if option["value"] == None:
                kernel_json["argv"].append(option["cmd"])
            else:
                kernel_json["argv"] += option["cmd"], str(option["value"])

    # other options from cmd line
    for option in cmd_opts:
        kernel_json["argv"].append(option)

    ksm = KernelSpecManager()

    # we need the installation dir of kernel, so first install a fake one
    tmp_kernel_name = "tmp_legion_kernel"
    tmp_kernel_json = {"argv": [], "display_name": "Tmp", "language": "python"}
    _install_kernel(ksm, tmp_kernel_name, tmp_kernel_json, user, prefix)
    spec = ksm.get_kernel_spec(tmp_kernel_name)
    kernel_install_dir = os.path.dirname(spec.resource_dir)
    _delete_kernel(ksm, tmp_kernel_name)

    # Now start installation
    kernel_name = cmd_dict["kernel_name"]

    # add installation dir to legin_kernel_nocr.py
    kernel_install_dir = os.path.join(kernel_install_dir, kernel_name)
    kernel_filename = kernel_json_suffix_nocr[0]
    kernel_json_suffix_nocr[0] = os.path.join(kernel_install_dir, kernel_filename)
    kernel_json["argv"] += kernel_json_suffix_nocr
    if verbose:
        print("The kernel_json is:\n" + str(kernel_json))

    # check if kernel is existed, if yes, then delete the old one before installation. 
    _delete_kernel(ksm, kernel_name, False)

    # install the kernel
    _install_kernel(ksm, kernel_name, kernel_json, user, prefix, False)

    # copy legion_kernel_nocr.py into kernel dir
    if kernel_file_dir == None:
        file_path = os.getcwd() + "/" + kernel_filename
    else:
        file_path = kernel_file_dir + "/" + kernel_filename
    shutil.copy(file_path, kernel_install_dir)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Install Legion IPython Kernel"
    )

    parser.add_argument(
        "--user",
        action="store_true",
        default=True,
        dest="user",
        help="Install the kernel in user home directory",
    )
    parser.add_argument(
        "--kernel-name",
        default="",
        dest="kernel_name",
        help="Install the kernel into prefix",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        dest="prefix",
        help="Install the kernel into prefix",
    )
    parser.add_argument(
        "--json",
        default="flexflow_python.json",
        dest="json",
        help="Configuration file of flexflow_python",
    )
    parser.add_argument(
        "--flexflow-python-prefix",
        default=None,
        dest="flexflow_python_prefix",
        help="The dirctory where flexflow_python is installed",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        dest="cpus",
        help="Number of CPUs to use per rank",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        dest="gpus",
        help="Number of GPUs to use per rank",
    )
    parser.add_argument(
        "--omps",
        type=int,
        default=0,
        dest="openmp",
        help="Number of OpenMP groups to use per rank",
    )
    parser.add_argument(
        "--ompthreads",
        type=int,
        default=4,
        dest="ompthreads",
        help="Number of threads per OpenMP group",
    )
    parser.add_argument(
        "--utility",
        type=int,
        default=1,
        dest="utility",
        help="Number of Utility processors per rank to request for meta-work",
    )
    parser.add_argument(
        "--sysmem",
        type=int,
        default=4000,
        dest="sysmem",
        help="Amount of DRAM memory per rank (in MBs)",
    )
    parser.add_argument(
        "--fbmem",
        type=int,
        default=4000,
        dest="fbmem",
        help="Amount of framebuffer memory per GPU (in MBs)",
    )
    parser.add_argument(
        "--zcmem",
        type=int,
        default=32,
        dest="zcmem",
        help="Amount of zero-copy memory per rank (in MBs)",
    )
    parser.add_argument(
        "--regmem",
        type=int,
        default=0,
        dest="regmem",
        help="Amount of registered CPU-side pinned memory per rank (in MBs)",
    )
    parser.add_argument(
        "--no-replicate",
        dest="not_control_replicable",
        action="store_true",
        required=False,
        default=True,
        help="Execute this program without control replication.  Most of the "
        "time, this is not recommended.  This option should be used for "
        "debugging.  The -lg:safe_ctrlrepl Legion option may be helpful "
        "with discovering issues with replicated control.",
    )
    parser.add_argument(
        "--launcher",
        dest="launcher",
        choices=["mpirun", "jsrun", "srun", "none"],
        default="none",
        help='launcher program to use (set to "none" for local runs, or if '
        "the launch has already happened by the time legate is invoked)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        dest="nodes",
        help="Number of nodes to use",
    )
    parser.add_argument(
        "--ranks-per-node",
        type=int,
        default=1,
        dest="ranks_per_node",
        help="Number of ranks (processes running copies of the program) to "
        "launch per node. The default (1 rank per node) will typically result "
        "in the best performance.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        dest="verbose",
        help="Display the detailed log of installation",
    )

    args, opts = parser.parse_known_args()
    return args, opts

def driver(args, opts, kernel_file_dir=None):
    cmd_dict = parse_json(flexflow_python_prefix=args.flexflow_python_prefix,
                          cpus=args.cpus,
                          gpus=args.gpus,
                          openmp=args.openmp,
                          ompthreads=args.ompthreads,
                          utility=args.utility,
                          sysmem=args.sysmem,
                          fbmem=args.fbmem,
                          zcmem=args.zcmem,
                          regmem=args.regmem,
                          launcher=args.launcher,
                          nodes=args.nodes,
                          ranks_per_node=args.ranks_per_node,
                          not_control_replicable=args.not_control_replicable,
                          kernel_name=args.kernel_name,
                          filename=args.json)

    if cmd_dict["not_control_replicable"]:
        install_kernel_nocr(user=args.user, 
                            prefix=args.prefix, 
                            cmd_opts=opts,
                            cmd_dict=cmd_dict,
                            verbose=args.verbose,
                            kernel_file_dir=kernel_file_dir)
    else:
        assert 0, "Control replication is not supported yet"

if __name__ == '__main__':
    args, opts = parse_args()
    driver(args, opts)
