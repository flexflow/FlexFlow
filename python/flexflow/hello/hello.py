from flexflow.core import *

import cffi
ffihello = cffi.FFI()
ffihello.cdef('void launch_hello_world_task(char *name);')
#hello_c = ffihello.dlopen("hello/libhello.so")
hello_c = ffihello.dlopen(None)

def launch_hello_world_task(name):
  args = BufferBuilder()
  args.pack_string(name)
  task = Task(task_id=112, data=args.get_string(), size=args.get_size())
  runtime = get_legion_runtime()
  context = get_legion_context()
  task.launch(runtime, context)

def launch_hello_world_task_c(name):
  c_name = ffi.new("char []", name.encode('utf-8'))
  #c_name = ffi.from_buffer(name.encode('utf-8'))
  hello_c.launch_hello_world_task(c_name)
