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

from __future__ import absolute_import, division, print_function, unicode_literals

import gc
import os
import sys
import code
import atexit
import readline
import threading

from .legion_cffi_header import ffi, lib as c


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

# This code is borrowed from the Python docs:
# https://docs.python.org/3/library/readline.html
class LegionConsole(code.InteractiveConsole):
  def __init__(self, locals=None, filename='<console>',
               histfile = os.path.expanduser('~/.python-history')):
    code.InteractiveConsole.__init__(self, locals, filename)
    self.init_history(histfile)

  def init_history(self, histfile):
    readline.parse_and_bind('tab: complete')
    if hasattr(readline, 'read_history_file'):
      try:
        readline.read_history_file(histfile)
      except FileNotFoundError:
        pass
      atexit.register(self.save_history, histfile)

  def save_history(self, histfile):
      readline.set_history_length(10000)
      readline.write_history_file(histfile)
     
def run_repl():
  try:
    shell = LegionConsole()
    shell.interact(banner='Welcome to Legion Python interactive console')
  except SystemExit:
    pass
    
def run_cmd(cmd, run_name=None):
    import imp
    module = imp.new_module(run_name)
    setattr(module, '__name__', run_name)
    setattr(module, '__package__', None)

    # Hide the current module if it exists.
    old_module = sys.modules.get(run_name)
    sys.modules[run_name] = module
    code = compile(cmd, '<string>', 'eval')
    exec(code, module.__dict__, module.__dict__)
    # Wait for execution to finish here before removing the module
    # because executing tasks might still need to refer to it
    future = c.legion_runtime_issue_execution_fence(
            top_level.runtime[0], top_level.context[0])
    # block waiting on the future
    c.legion_future_wait(future, True, ffi.NULL)
    c.legion_future_destroy(future)
    # Make sure our module gets deleted to clean up any references
    # to variables the user might have made
    if old_module is None:
        del sys.modules[run_name]
    else:
        sys.modules[run_name] = old_module
    del module

# We can't use runpy for this since runpy is aggressive about
# cleaning up after itself and removes the module before execution
# has completed.
def run_path(filename, run_name=None):
    import types
    module = types.ModuleType(run_name)
    setattr(module, '__name__', run_name)
    setattr(module, '__file__', filename)
    setattr(module, '__loader__', None)
    setattr(module, '__package__', run_name.rpartition('.')[0])

    # Hide the current module if it exists.
    old_module = sys.modules.get(run_name)
    sys.modules[run_name] = module

    sys.path.append(os.path.dirname(filename))

    with open(filename) as f:
      code = compile(f.read(), filename, 'exec')
      exec(code, module.__dict__, module.__dict__)
    # Wait for execution to finish here before removing the module
    # because executing tasks might still need to refer to it
    future = c.legion_runtime_issue_execution_fence(
            top_level.runtime[0], top_level.context[0])
    # block waiting on the future
    c.legion_future_wait(future, True, ffi.NULL)
    c.legion_future_destroy(future)
    # Make sure our module gets deleted to clean up any references
    # to variables the user might have made
    if old_module is None:
      del sys.modules[run_name]
    else:
      sys.modules[run_name] = old_module
    del module



def flexflow_top_level_task(raw_args, user_data, proc):
    print("start top-level task")
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
    start = 1
    if len(args) > 1 and args[1] == '--nocr':
      start += 1
    if len(args) < (start+1) or args[start] == '-':
      run_repl()
    elif args[start] == '-c':
      assert len(args) >= 3
      sys.argv = list(args)
      run_cmd(args[start+1], run_name='__main__')
    else:
      assert len(args) >= (start+1) 
      sys.argv = list(args)
      run_path(args[start], run_name='__main__')

    future = c.legion_runtime_issue_execution_fence(runtime[0], context[0])
    c.legion_future_wait(future, False, ffi.NULL)
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
    print("real end top-level task")
    
def get_legion_runtime():
    return top_level.runtime

def get_legion_context():
    return top_level.context
