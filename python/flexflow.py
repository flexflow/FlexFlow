#!/usr/bin/env python

from __future__ import print_function
import argparse, json, os, platform, subprocess, sys

_version = sys.version_info.major

if _version == 3: # Python 3.x:
    _input = input
else:
    raise Exception('Incompatible Python version')

os_name = platform.system()

if os_name == 'Linux':
    dylib_ext = '.so'
elif os_name == 'Darwin': # Don't currently support Darwin at the moment
    dylib_ext = '.dylib'
else:
    raise Exception('FlexFlow does not work on %s' % platform.system())

def run_flexflow(nodes, cpus, gpus, utility, fbmem, zcmem, opts):
    # Start building our command line for the subprocess invocation
    cmd_env = dict(os.environ.items())
    if nodes > 1:
        cmd = ['mpirun', '-n', str(nodes), '--npernode', '1', '--bind-to', 'none', '--mca', 'mpi_warn_on_fork', '0']
    else:
        cmd = []
        cmd = ['mpirun', '-n', str(nodes), '--npernode', '1', '--bind-to', 'none', '--mca', 'mpi_warn_on_fork', '0']
    # Set the path to the FlexFlow module as an environment variable
    flexflow_python_dir = os.path.dirname(os.path.realpath(__file__))
    flexflow_dir = os.path.abspath(os.path.join(flexflow_python_dir, os.pardir))
    if 'PYTHON_MODULES_PATH' in cmd_env:
        cmd_env['PYTHON_MODULES_PATH'] += os.pathsep + flexflow_python_dir 
    else:
        cmd_env['PYTHON_MODULES_PATH'] = flexflow_python_dir
    if nodes > 1:
        cmd += ['-x', 'PYTHON_MODULES_PATH']

    # Always make sure we include the Legion runtime dir and FlexFlow dir
    if 'LG_RT_DIR' not in os.environ:
        legion_dir = os.path.join(flexflow_dir, 'legion')
        runtime_dir = os.path.join(legion_dir, 'runtime')
        cmd_env['LG_RT_DIR'] = runtime_dir
        
    if 'FF_HOME' not in os.environ:
        cmd_env['FF_HOME'] = flexflow_dir
        
    if nodes > 1:
        cmd += ['-x', 'LG_RT_DIR']
        cmd += ['-x', 'FF_HOME']

    # We always need one python processor per node and no local fields per node
    cmd += [str(os.path.join(flexflow_python_dir, 'flexflow_python')), '-ll:py', '1']

    if cpus != 1:
        cmd += ['-ll:cpu', str(cpus)]
    if gpus > 0:
        cmd += ['-ll:gpu', str(gpus)]
        cmd += ['-ll:fsize', str(fbmem), '-ll:zsize', str(zcmem)]
    if utility != 1:
        cmd += ['-ll:util', str(utility)]

    # Now we can append the result of the flags to the command
    if opts:
        cmd += opts
    
    print(cmd)
    # Launch the child process
    child_proc = subprocess.Popen(cmd, env = cmd_env)
    # Wait for it to finish running
    result = child_proc.wait()
    return result

def driver():
    parser = argparse.ArgumentParser(
        description='Legate Driver.')
    parser.add_argument(
            '--nodes', type=int, default=1, dest='nodes', help='Number of nodes to use')
    parser.add_argument(
            '--cpus', type=int, default=1, dest='cpus', help='Number of CPUs per node to use')
    parser.add_argument(
            '--gpus', type=int, default=1, dest='gpus', help='Number of GPUs per node to use')
    parser.add_argument(
            '--utility', type=int, default=1, dest='utility', help='Number of Utility processors per node to request for meta-work')
    parser.add_argument(
            '--fbmem', type=int, default=2048, dest='fbmem', help='Amount of framebuffer memory per GPU (in MBs)')
    parser.add_argument(
            '--zcmem', type=int, default=12192, dest='zcmem', help='Amount of zero-copy memory per node (in MBs)')
    args, opts = parser.parse_known_args()
    # See if we have at least one script file to run
    console = True
    for opt in opts:
        if '.py' in opt:
            console = False
            break
    assert console == False, "Please provide a python file"
    return run_flexflow(args.nodes, args.cpus, args.gpus, args.utility, args.fbmem, args.zcmem, opts)

if __name__ == '__main__':
    sys.exit(driver())

