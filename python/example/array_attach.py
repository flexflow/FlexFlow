from flexflow.core import *

import numpy as np


def arr_from_ptr(pointer, typestr, shape, copy=False,
                 read_only_flag=False):
  buff = {'data': (pointer, read_only_flag),
          'typestr': typestr,
          'shape': shape}

  class numpy_holder():
    pass

  holder = numpy_holder()
  holder.__array_interface__ = buff
  return np.array(holder, copy=copy)
    
def top_level_task():

  arr = np.ones(10, dtype=np.float32)
  pointer, read_only_flag = arr.__array_interface__['data']
  arr_out = arr_from_ptr(pointer, '<f4', (10,))
  pointer, read_only_flag = arr_out.__array_interface__['data']
  print(pointer)
  arr_out = arr_out + 1.1
  pointer, read_only_flag = arr_out.__array_interface__['data']
  print(pointer)
  #assert np.allclose(arr, arr_out)
  
  # base_ptr = malloc_int(6*6)
  # base_ptr_int = int(ffi.cast("uintptr_t", base_ptr))
  # print(base_ptr_int)
  # shape = (6, 6)
  # arr_from_ptr(base_ptr_int, '<f4', (10,))
  
  base_ptr = malloc_int(6*6)
  base_ptr_int = int(ffi.cast("uintptr_t", base_ptr))
  print(base_ptr_int)
  shape = (6, 6)
  print(shape)
  strides = None
  initializer = RegionNdarray(shape, "i", base_ptr_int, strides, False)
  array = np.asarray(initializer)
  array[1,1] = 10
  pointer, read_only_flag = array.__array_interface__['data']
  print(array)
  print(pointer)
  print_array_int(base_ptr, 6*6)

if __name__ == "__main__":
  top_level_task()