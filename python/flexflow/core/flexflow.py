#!/usr/bin/env python

# Copyright 2020 Stanford University
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

from __future__ import absolute_import, division, print_function, unicode_literals

import gc
import os
import sys
import threading

from flexflow.core.legion import *
from flexflow.core.cbinding import *
from flexflow.core.legion_cffi import ffi, lib as c


# Storage for variables that apply to the top-level task.
# IMPORTANT: They are valid ONLY in the top-level task.
top_level = threading.local()
# Fields:
#     top_level.runtime
#     top_level.context
#     top_level.task
#     top_level.cleanup_items


def input_args(filter_runtime_options=False):
    raw_args = c.legion_runtime_get_input_args()

    args = []
    for i in range(raw_args.argc):
        args.append(ffi.string(raw_args.argv[i]).decode('utf-8'))

    if filter_runtime_options:
        i = 1 # Skip program name

        prefixes = ['-lg:', '-hl:', '-realm:', '-ll:', '-cuda:', '-numa:',
                    '-dm:', '-bishop:']
        while i < len(args):
            match = False
            for prefix in prefixes:
                if args[i].startswith(prefix):
                    match = True
                    break
            if args[i] == '-level':
                match = True
            if args[i] == '-logfile':
                match = True
            if match:
                args.pop(i)
                # Assume that every option has an argument, as long as
                # the subsequent value does **NOT** start with a dash.
                if i < len(args) and not args[i].startswith('-'):
                    args.pop(i)
                continue
            i += 1
    return args


# We can't use runpy for this since runpy is aggressive about
# cleaning up after itself and removes the module before execution
# has completed.
def run_path(filename, run_name=None):
    import imp
    module = imp.new_module(run_name)
    setattr(module, '__name__', run_name)
    setattr(module, '__file__', filename)
    setattr(module, '__loader__', None)
    setattr(module, '__package__', run_name.rpartition('.')[0])

    # Hide the current module if it exists.
    old_module = sys.modules[run_name] if run_name in sys.modules else None
    sys.modules[run_name] = module

    sys.path.append(os.path.dirname(filename))

    with open(filename) as f:
        code = compile(f.read(), filename, 'exec')
        exec(code, module.__dict__)

    # FIXME: Can't restore the old module because tasks may be
    # continuing to execute asynchronously. We could fix this with
    # an execution fence but it doesn't seem worth it given that
    # we'll be cleaning up the process right after this.

    # sys.modules[run_name] = old_module


def flexflow_top_level_task(raw_args, user_data, proc):
    raw_arg_ptr = ffi.new('char[]', bytes(raw_args))
    raw_arg_size = len(raw_args)

    # Execute preamble to obtain Legion API context.
    task = ffi.new('legion_task_t *')
    raw_regions = ffi.new('legion_physical_region_t **')
    num_regions = ffi.new('unsigned *')
    context = ffi.new('legion_context_t *')
    runtime = ffi.new('legion_runtime_t *')
    c.legion_task_preamble(
        raw_arg_ptr, raw_arg_size, proc,
        task, raw_regions, num_regions, context, runtime)

    top_level.runtime, top_level.context, top_level.task = runtime, context, task
    top_level.cleanup_items = []

    print("top-level task")
    # Run user's script.
    args = input_args(True)
    assert len(args) >= 2
    sys.argv = list(args)
    run_path(args[1], run_name='__main__')
    print("end top-level task")

    # # Hack: Keep this thread alive because otherwise Python will reuse
    # # it for task execution and Pygion's thread-local state (_my.ctx)
    # # will get messed up.
    # c.legion_future_get_void_result(
    #     c.legion_runtime_issue_execution_fence(runtime[0], context[0]))

    for cleanup in top_level.cleanup_items:
        cleanup()

    del top_level.runtime
    del top_level.context
    del top_level.task
    del top_level.cleanup_items
    
    gc.collect()

    # Execute postamble.
    c.legion_task_postamble(runtime[0], context[0], ffi.NULL, 0)
    #print("real end top-level task")
    
def get_legion_runtime():
    return top_level.runtime

def get_legion_context():
    return top_level.context
