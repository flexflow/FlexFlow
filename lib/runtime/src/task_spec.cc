#include "task_spec.h"
#include "operator.h"
#include "utils/containers.h"

using namespace Legion;

namespace FlexFlow {

TensorSpec::TensorSpec(TensorRole tensor_role, int idx, IsGrad is_grad, optional<Legion::PrivilegeMode> mode)
  : role(tensor_role), idx(idx), is_grad(is_grad), mode(mode)
{ }

OpTaskSpec::OpTaskSpec(TaskID task_id, OpTaskType task_type) 
  : task_id(task_id), task_type(task_type)
{ }

void OpTaskSpec::add_input_slot(int slot) {
  this->slots.insert({slot, TensorRole::INPUT});
}

void OpTaskSpec::add_param_slot(int slot) {
  this->slots.insert({slot, TensorRole::PARAM});
}

void OpTaskSpec::add_output_slot(int slot) {
  this->slots.insert({slot, TensorRole::OUTPUT});
}

static bool spec_satisfies_slot_role(TensorRole slot_role, TensorSpec const &tensor_spec) {
  if (slot_role == TensorRole::INPUT || slot_role == TensorRole::PARAM) {
    return (tensor_spec.mode == READ_ONLY || tensor_spec.mode == READ_WRITE);
  } else if (slot_role == TensorRole::OUTPUT) {
    return (tensor_spec.mode == WRITE_ONLY || tensor_spec.mode == READ_WRITE);
  }
}

void OpTaskSpec::bind(int slot, TensorSpec const &tensor_spec) {
  assert (contains_key(this->slots, slot));
  assert (spec_satisfies_slot_role(this->slots.at(slot), tensor_spec));

  if (!contains_l(this->region_idxs, tensor_spec)) {
    region_idxs.equate(tensor_spec, this->new_region_idx());  
  }

  this->bindings.insert({slot, tensor_spec});
}

void OpTaskSpec::bind(std::vector<std::pair<int, TensorSpec>> const &bindings) {
  for (auto const &binding : bindings) {
    this->bind(binding.first, binding.second);
  }
}

tl::optional<TensorSpec const &> OpTaskSpec::in_slot(int slot) const {
  return this->bindings.at(slot);
}

int OpTaskSpec::get_region_idx(TensorSpec const &tensor_spec) const {
  return this->region_idxs.at_l(tensor_spec);
};

optional<int> OpTaskSpec::get_region_idx(int slot) const {
  auto tensor_spec = this->in_slot(slot);
  if (!tensor_spec.has_value()) {
    return nullopt;
  } else {
    return this->get_region_idx(tensor_spec.value());
  }
}

int OpTaskSpec::new_region_idx() {
  int result = this->region_idx_counter;
  this->region_idx_counter++;
  return result;
}

TaskAccessor(Task const *task, std::vector<Legion::PhysicalRegion> const &regions, Legion::Context const &ctx, Legion::Runtime *runtime, TaskSpec const &spec)
  : task(task), regions(regions), ctx(ctx), runtime(runtime), spec(spec)
{ }

TaskAccessor(Task const *task, std::vector<Legion::PhysicalRegion> const &regions, Legion::Context const &ctx, Legion::Runtime *runtime, OpTaskType task_type)
  : TaskAccessor(task, regions, ctx, runtime, ((Op const *)task->args)->get_tasks_spec(task_type))
{ }

}
