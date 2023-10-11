#include "sim_environment.h"

namespace FlexFlow {

// SimTaskBinding
void SimTaskBinding::bind(slot_id id, ParallelTensorShape const &shape) {
  InputParallelTensorDesc desc = {shape, IsTrainable::YES};
  this->tensor_shape_bindings.insert({id, desc});
}

void SimTaskBinding::bind_untrainable(slot_id id, ParallelTensorShape const &shape) {
  InputParallelTensorDesc desc = {shape, IsTrainable::NO};
  this->tensor_shape_bindings.insert({id, desc});
}

void SimTaskBinding::bind(slot_id id, ParallelTensorShape const &shape, IsTrainable trainable) {
  InputParallelTensorDesc desc = {shape, trainable};
  this->tensor_shape_bindings.insert({id, desc});
}

void SimTaskBinding::bind(slot_id id, InputParallelTensorDesc const & desc) {
  this->tensor_shape_bindings.insert({id, desc});
}

void SimTaskBinding::bind(slot_id id, std::vector<ParallelTensorShape> const &shapes) {
  InputVariadicParallelTensorDesc desc = {shapes, IsTrainable::YES};
  this->tensor_shape_bindings.insert({id, desc});
}

void SimTaskBinding::bind_untrainable(slot_id id, std::vector<ParallelTensorShape> const &shapes) {
  InputVariadicParallelTensorDesc desc = {shapes, IsTrainable::NO};
  this->tensor_shape_bindings.insert({id, desc});
}

void SimTaskBinding::bind(slot_id id, std::vector<ParallelTensorShape> const & shapes, IsTrainable trainable) {
  InputVariadicParallelTensorDesc desc = {shapes, trainable};
  this->tensor_shape_bindings.insert({id, desc});
}

void SimTaskBinding::bind(slot_id id, InputVariadicParallelTensorDesc const & desc) {
  this->tensor_shape_bindings.insert({id, desc});
}

template <typename T>
void SimTaskBinding::bind_arg(slot_id name, OpArgRef<T> const &ref) {
  this->insert_arg_spec(name, OpArgRefSpec::create(ref));
}

// SimEnvironment
TaskArgumentAccessor SimEnvironment::get_init_accessor(
    task_id_t tid, SimTaskBinding const &sim_task_binding) {
  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(get_op_signature(tid), sim_task_binding);
}

TaskArgumentAccessor SimEnvironment::get_fwd_accessor(
    task_id_t tid, SimTaskBinding const &sim_task_binding) {
  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(get_op_signature(tid), sim_task_binding);
}

TaskArgumentAccessor SimEnvironment::get_bwd_accessor(
    task_id_t tid, SimTaskBinding const &sim_task_binding) {
  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(get_op_signature(tid), sim_task_binding);
}

// Sim Allocation


} // namespace FlexFlow
