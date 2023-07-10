from .raw import flexflow_computation_graph_create, flexflow_computation_graph_t, check_type, flexflow_tensor_create, flexflow_tensor_t, flexflow_datatype_t, flexflow_tensor_get_create_grad, flexflow_tensor_get_datatype
from flexflow.op_attrs import DataType, ParameterSyncType
from typing import Optional, List

class Tensor:
  def __init__(self, handle)
    check_type(handle, flexflow_tensor_t)
    self._handle = handle 

  @property
  def create_grad(self) -> bool:
    return flexflow_tensor_get_create_grad(self._handle)

  @property
  def datatype(self) -> bool:
    return flexflow_tensor_get_datatype(self._handle)

  @property
  def sync_type(self) -> ParameterSyncType:
    return flexflow_tensor_get_sync_type(self._handle)

  @property
  def initializer(self) -> Initializer:
    return flexflow_tensor_get_initializer(self._handle)

  @property
  def dims(self) -> Tuple[int,...]:
    return flexflow_tensor_get_dims(self._handle)

  @classmethod
  def new(self,
          comp_graph: ComputationGraph,
          dims: List[int], 
          datatype: DataType,
          sync_type: ParameterSyncType):
    return Tensor(flexflow_tensor_create(comp_graph, dims, datatype, sync_type))

class ComputationGraph:
  def __init__(self):
    self._handle = flexflow_computation_graph_create()
    check_type(self._handle, flexflow_computation_graph_t)

  def exp(self, input_tensor: Tensor, name: Optional[str] = None) -> Tensor:
    return Tensor(flexflow_computation_graph_add_op_exp(self._handle, input_tensor._handle, name))

  def add(self, input_lhs: Tensor, input_rhs: Tensor, name: Optional[str] = None) -> Tensor:
    return Tensor(flexflow_computation_graph_add_op_add(self._handle, input_lhs._handle, input_rhs._handle, name))

  def subtract(self, input_lhs: Tensor, input_rhs: Tensor, name: Optional[str] = None) -> Tensor:
    return Tensor(flexflow_computation_graph_add_op_subtract(self._handle, input_lhs._handle, input_rhs._handle, name))
