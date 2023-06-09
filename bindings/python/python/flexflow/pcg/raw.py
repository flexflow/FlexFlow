from .flexflow_cffi_header import ffc, ffi
from enum import Enum
from .utils import handle_error_code
from flexflow.op_attrs import ParameterSyncType

flexflow_computation_graph_t = ffi.typeof('flexflow_computation_graph_t')
flexflow_tensor_t = ffi.typeof('flexflow_tensor_t')
flexflow_initializer_t = ffi.typeof('flexflow_initializer_t')
flexflow_param_sync_t = ffi.typeof('flexflow_param_sync_t')
bool_t = ffi.typeof('bool')

def optional(t):
  return (t, typeof(None))

def unwrap_ptr(v):
  if v is None:
    return ffi.NULL
  else:
    return v

def check_returned(ptr, ty):
  assert isinstance(ptr[0], ty)
  return ptr[0]

def ptr_to(t):
  if t is flexflow_computation_graph_t:
    return ffi.typeof('flexflow_computation_graph_t *')
  elif t is flexflow_tensor_t:
    return ffi.typeof('flexflow_tensor_t *')
  else:
    raise ValueError(f'Unknown ffi.typeof {t}')

def valid_ptr_to(p):
  raise NotImplementedError

def list_of(p):
  raise NotImplementedError

def check_type(value, ty):
  if ffi.typeof(value) is not ty:
    raise TypeError(f'Value {value} does not have type {ty}')

def allocate_new_computation_graph():
  ptr = ffi.new('flexflow_computation_graph_t *')
  assert ptr != ffi.NULL  # I don't know if this is possible if allocation fails
  return ptr

@handle_error_code
def _flexflow_computation_graph_create(comp_graph_ptr):
  check_type(comp_graph_ptr, valid_ptr_to(flexflow_computation_graph_t))
  return ffc.flexflow_computation_graph_create(comp_graph_ptr)

@handle_error_code
def _flexflow_computation_graph_destroy(comp_graph):
  check_type(comp_graph, flexflow_computation_graph_t)
  return ffc.flexflow_computation_graph_destroy(comp_graph)

def flexflow_computation_graph_create():
  comp_graph_ptr = allocate_new_computation_graph()
  _flexflow_computation_graph_create(comp_graph_ptr)
  managed = ffi.gc(comp_graph_ptr, lambda ptr: _flexflow_computation_graph_destroy(ptr[0]))
  return managed[0]

def allocate_new_tensor():
  ptr = ffi.new('flexflow_tensor_t *')
  assert ptr != ffi.NULL
  return ptr

@handle_error_code
def _flexflow_tensor_create(comp_graph, num_dims, dims, datatype, create_grad, out_ptr):
  check_type(comp_graph, computation_graph_t)
  check_type(num_dims, int)
  check_type(dims, list_of(int))
  if len(dims) != num_dims:
    raise ValueError(f'num_dims {num_dims} does not match length of dims {dims}')
  check_type(out_ptr, valid_ptr_to(flexflow_tensor_t))
  return ffc.flexflow_tensor_create(comp_graph, num_dims, dims, datatype, create_grad, out_ptr)

@handle_error_code
def _flexflow_tensor_get_create_grad(tensor, out):
  check_type(tensor, flexflow_tensor_t)
  check_type(out, ptr_to(bool_t))
  return ffc.flexflow_tensor_get_create_grad(tensor, out)

def flexflow_tensor_get_create_grad(tensor) -> bool:
  ptr = ffi.new('bool *')
  _flexflow_tensor_get_create_grad(tensor, ptr)
  return 

@handle_error_code
def _flexflow_tensor_get_initializer(tensor, out):
  check_type(tensor, flexflow_tensor_t)
  check_type(out, ptr_to(flexflow_initializer_t))
  return ffc.flexflow_tensor_get_initializer(tensor, out)

@handle_error_code
def _flexflow_tensor_get_sync_type(tensor, out):
  check_type(tensor, flexflow_tensor_t)
  check_type(out, ptr_to(flexflow_param_sync_t))
  return ffc.flexflow_tensor_get_sync_type(tensor, out)

def flexflow_tensor_get_sync_type(tensor) -> ParameterSyncType:
  ptr = ffi.new('flexflow_param_sync_t *')
  _flexflow_tensor_get_sync_type(tensor, ptr)
  return ParameterSyncType(check_returned(ptr, int))

def _flexflow_tensor_get_num_dims(tensor, out):
  check_type(tensor, flexflow_tensor_t)
  check_type(out, ptr_to(int))
  return ffc.flexflow_tensor_get_num_dims(tensor, out)

def _flexflow_tensor_get_dims(tensor, out):
  check_type(tensor, flexflow_tensor_t)
  check_type(out, ptr_to(int))
  return ffc.flexflow_tensor_get_dims(tensor, out)

def flexflow_tensor_get_num_dims(tensor) -> int:
  ptr = ffi.new('int *')
  _flexflow_tensor_get_num_dims(tensor, ptr)
  return check_returned(ptr, int)

def flexflow_tensor_get_dims(tensor) -> Tuple[int,...]:
  num_dims = flexflow_tensor_get_num_dims(tensor)
  assert num_dims >= 1
  dims_arr = ffi.new(f'int[{num_dims}]')
  _flexflow_tensor_get_dims(ffi.cast('int*', dims))
  result = tuple(dims_arr[i] for i in range(num_dims))
  return result

@handle_error_code
def _flexflow_tensor_get_datatype(tensor, out):
  check_type(tensor, flexflow_tensor_t)
  check_type(out, ptr_to(flexflow_datatype_t))
  return ffc.flexflow_tensor_get_datatype(tensor, out)

def flexflow_tensor_get_datatype(tensor) -> DataType:
  ptr = ffi.new('flexflow_datatype_t *')
  _flexflow_tensor_get_datatype(tensor, ptr)
  return DataType(check_returned(ptr, int))

@handle_error_code
def _flexflow_tensor_destroy(comp_graph, tensor)
  check_type(comp_graph, flexflow_computation_graph_t)
  check_type(tensor, flexflow_tensor_t)
  return ffc.flexflow_tensor_destroy(comp_graph, tensor)

def allocate_new_tensor_gc():
  ptr = allocate_new_tensor()
  managed = ffi.gc(tensor_ptr, lambda ptr: _flexflow_tensor_destroy(ptr[0])) # TODO FIXME @lockshaw: is this really necessary, or is comp_graph responsible for cleanup? should probably be comp_graph
  return ptr

def flexflow_tensor_create(comp_graph, dims, datatype, create_grad):
  tensor_ptr = allocate_new_tensor_gc()
  _flexflow_tensor_create(comp_graph, len(dims), dims, datatype, create_grad, tensor_ptr)
  return managed[0]

@handle_error_code
def _flexflow_computation_graph_add_op_exp(comp_graph, tensor, out_ptr, name):
  check_type(comp_graph, flexflow_computation_graph_t)
  check_type(tensor, flexflow_tensor_t)
  check_type(out_ptr, valid_ptr_to(flexflow_tensor_t))
  check_type(name, optional(str))
  return ffc.flexflow_computation_graph_add_op_exp(comp_graph, tensor, out_ptr, unwrap_ptr(name))

def flexflow_computation_graph_add_op_exp(comp_graph, tensor, name) -> Tensor:
  output_tensor_ptr = allocate_new_tensor_gc()
  _flexflow_computation_graph_add_op_exp(comp_graph, tensor, output_tensor_ptr, name)
  return check_returned(output_tensor_ptr, flexflow_tensor_t)

@handle_error_code
def _flexflow_computation_graph_add_op_add(comp_graph, tensor_lhs, tensor_rhs, out_ptr, name):
  check_type(comp_graph, flexflow_computation_graph_t)
  check_type(tensor_lhs, flexflow_tensor_t)
  check_type(tensor_rhs, flexflow_tensor_t)
  check_type(out_ptr, valid_ptr_to(flexflow_tensor_t))
  check_type(name, optional(str))
  return ffc.flexflow_computation_graph_add_op_add(comp_graph, tensor_lhs, tensor_rhs, out_ptr, unwrap_ptr(name))

def flexflow_computation_graph_add_op_add(comp_graph, lhs_tensor, rhs_tensor, name):
  output_tensor_ptr = allocate_new_tensor_gc()
  _flexflow_computation_graph_add_op_add(comp_graph, lhs_tensor, rhs_tensor, output_tensor_ptr, name)
  return check_returned(output_tensor_ptr, flexflow_tensor_t)

@handle_error_code
def _flexflow_computation_graph_add_op_subtract(comp_graph, tensor_lhs, tensor_rhs, out_ptr, name):
  check_type(comp_graph, flexflow_computation_graph_t)
  check_type(tensor_lhs, flexflow_tensor_t)
  check_type(tensor_rhs, flexflow_tensor_t)
  check_type(out_ptr, valid_ptr_to(flexflow_tensor_t))
  check_type(name, optional(str))
  return ffc.flexflow_computation_graph_add_op_subtract(comp_graph, tensor_lhs, tensor_rhs, out_ptr, unwrap_ptr(name))

def flexflow_computation_graph_add_op_subtract(comp_graph, lhs_tensor, rhs_tensor, name) -> Tensor:
  output_tensor_ptr = allocate_new_tensor_gc()
  _flexflow_computation_graph_add_op_subtract(comp_graph, lhs_tensor, rhs_tensor, name)
  return check_returned(output_tensor_ptr, flexflow_tensor_t)
