#include "sim_environment.h"

namespaec FlexFlow {

  void SimTaskBinding::bind(slot_id id, ParallelTensorShape const &shape) {
    tensor_shape_bindings.insert(id, shape);
  }
  void SimTaskBinding::bind(slot_id id, TensorShape const &shape) {
    tensor_shape_bindings.insert(id, shape);
  }

  void SimTaskBinding::bind(slot_id id,
                            InputVariadicParallelTensorDesc const &desc) {
    this->tensor_shape_bindings.insert(id, desc);
  }

  void SimTaskBinding::bind_arg(slot_id id, SimArg const &arg) {
    arg_bindings.insert(id, arg);
  }

  TaskArgumentAccessor SimEnvironment::get_fwd_accessor(
      task_id_t tid, SimTaskBinding const &sim_task_binding) {
    NOT_IMPLEMENTED(); // TODO
  }

} // namespace FlexFlow