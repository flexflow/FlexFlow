from flexflow import *

import cffi
ffihello = cffi.FFI()
ffihello.cdef('void launch_hello_world_task();')
#hello_c = ffihello.dlopen("hello/libhello.so")
hello_c = ffihello.dlopen(None)

def launch_hello_world_task():
  task = Task(task_id=112)
  runtime = get_legion_runtime()
  context = get_legion_context()
  task.launch(runtime, context)

if __name__ == '__main__':
  print("start hello")
  launch_hello_world_task()
  hello_c.launch_hello_world_task()
