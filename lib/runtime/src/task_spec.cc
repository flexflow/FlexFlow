#include "task_spec.h"
#include "operator.h"
#include "utils/containers.h"

using namespace Legion;

namespace FlexFlow {

Legion::PrivilegeMode get_default_fwd_privs(TensorRole tensor_role, IsGrad is_grad) {
  assert (is_grad == IsGrad::NO); 
  
  if (tensor_role == TensorRole::INPUT || tensor_role == TensorRole::PARAM) {
    return READ_ONLY;
  } else {
    return WRITE_ONLY;
  }
}

Legion::PrivilegeMode get_default_bwd_privs(TensorRole tensor_role, IsGrad is_grad) {
  if (tensor_role == TensorRole::INPUT && is_grad == IsGrad::YES) {
    return READ_WRITE;
  } else if (tensor_role == TensorRole::INPUT && is_grad == IsGrad::NO) {
    return READ_ONLY;
  } else if (tensor_role == TensorRole::PARAM && is_grad == IsGrad::YES) {

  }
}

Legion::PrivilegeMode TensorSlotSpec::get_privileges(OpTaskType task_type) const {
  if (this->mode.has_value()) {
    return this->mode.value();
  }

  switch (task_type) {
    case FWD:
  };
}

TensorSlotSpec get_backward_slot(TensorSlotSpec const &forward_slot) {
  assert (forward_slot.is_grad == IsGrad::NO);
  return {
    forward_slot.name,
    forward_slot.slot_type,
    forward_slot.tensor_role,
    IsGrad::NO
  };
}

TensorSlotSpec get_backward_grad_slot(TensorSlotSpec const &forward_slot) {
  return {
    forward_slot.name,
    forward_slot.slot_type,
    forward_slot.tensor_role,
    IsGrad::YES
  };
}

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd) {
  OpTaskSignature bwd(OpTaskType::BWD);

  for (SlotSpec const &slot : fwd.get_slots()) {
    if (is_tensor_slot(slot)) {
      TensorSlotSpec tensor_slot = get_tensor_slot(slot);
      assert (tensor_slot.is_grad == IsGrad::NO);
    }
  }
}


/* TensorSpec::TensorSpec(TensorRole tensor_role, int idx, IsGrad is_grad, optional<Legion::PrivilegeMode> mode) */
/*   : role(tensor_role), idx(idx), is_grad(is_grad), mode(mode) */
/* { } */

/* OpTaskSpec::OpTaskSpec(TaskID task_id, OpTaskType task_type) */ 
/*   : task_id(task_id), task_type(task_type) */
/* { } */

/* void OpTaskSpec::add_input_slot(int slot) { */
/*   this->slots.insert({slot, TensorRole::INPUT}); */
/* } */

/* void OpTaskSpec::add_param_slot(int slot) { */
/*   this->slots.insert({slot, TensorRole::PARAM}); */
/* } */

/* void OpTaskSpec::add_output_slot(int slot) { */
/*   this->slots.insert({slot, TensorRole::OUTPUT}); */
/* } */

/* static bool spec_satisfies_slot_role(TensorRole slot_role, TensorSpec const &tensor_spec) { */
/*   if (slot_role == TensorRole::INPUT || slot_role == TensorRole::PARAM) { */
/*     return (tensor_spec.mode == READ_ONLY || tensor_spec.mode == READ_WRITE); */
/*   } else if (slot_role == TensorRole::OUTPUT) { */
/*     return (tensor_spec.mode == WRITE_ONLY || tensor_spec.mode == READ_WRITE); */
/*   } */
/* } */

/* void OpTaskSpec::bind(int slot, TensorSpec const &tensor_spec) { */
/*   assert (contains_key(this->slots, slot)); */
/*   assert (spec_satisfies_slot_role(this->slots.at(slot), tensor_spec)); */

/*   if (!contains_l(this->region_idxs, tensor_spec)) { */
/*     region_idxs.equate(tensor_spec, this->new_region_idx()); */  
/*   } */

/*   this->bindings.insert({slot, tensor_spec}); */
/* } */

/* void OpTaskSpec::bind(std::vector<std::pair<int, TensorSpec>> const &bindings) { */
/*   for (auto const &binding : bindings) { */
/*     this->bind(binding.first, binding.second); */
/*   } */
/* } */

/* tl::optional<TensorSpec const &> OpTaskSpec::in_slot(int slot) const { */
/*   return this->bindings.at(slot); */
/* } */

/* int OpTaskSpec::get_region_idx(TensorSpec const &tensor_spec) const { */
/*   return this->region_idxs.at_l(tensor_spec); */
/* }; */

/* optional<int> OpTaskSpec::get_region_idx(int slot) const { */
/*   auto tensor_spec = this->in_slot(slot); */
/*   if (!tensor_spec.has_value()) { */
/*     return nullopt; */
/*   } else { */
/*     return this->get_region_idx(tensor_spec.value()); */
/*   } */
/* } */

/* int OpTaskSpec::new_region_idx() { */
/*   int result = this->region_idx_counter; */
/*   this->region_idx_counter++; */
/*   return result; */
/* } */

/* TaskAccessor(Task const *task, std::vector<Legion::PhysicalRegion> const &regions, Legion::Context const &ctx, Legion::Runtime *runtime, TaskSpec const &spec) */
/*   : task(task), regions(regions), ctx(ctx), runtime(runtime), spec(spec) */
/* { } */

/* TaskAccessor(Task const *task, std::vector<Legion::PhysicalRegion> const &regions, Legion::Context const &ctx, Legion::Runtime *runtime, OpTaskType task_type) */
/*   : TaskAccessor(task, regions, ctx, runtime, ((Op const *)task->args)->get_tasks_spec(task_type)) */
/* { } */

}
