#include "operator.h"
#include "op-meta/ffconst_utils.h"
#include <stdexcept>

using namespace Legion;

namespace FlexFlow {

TensorSpec::TensorSpec(TensorRole role, int idx, IsGrad is_grad, tl::optional<PrivilegeMode> mode)
  : role(role), idx(idx), is_grad(is_grad), mode(mode) 
{ }

size_t Op::get_untyped_params_hash() const {
  size_t hash = this->get_params_hash();
  hash_combine(hash, this->op_type);
  return hash;
}

size_t Op::get_params_hash() const {
  throw std::runtime_error(
      "No overload of get_params_hash defined for op type " +
      get_operator_type_name(this->op_type));
}

void Op::require_input_tensor(IndexLauncher &launcher, int idx) const {
  launcher.add_region_requirement(RegionRequirement(this->inputs.at(idx)->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    this->inputs.at(idx)->region));
}

void Op::require_output_tensor(Legion::IndexLauncher &launcher, int idx) const {
  launcher.add_region_requirement(RegionRequirement(this->outputs.at(idx)->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    this->outputs.at(idx)->region));
}

void Op:require_weight_tensor(Legion::IndexLauncher &launcher, int idx) const {
  launcher.add_region_requirement(RegionRequirement(this->weights.at(idx)->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    this->weights.at(idx)->region));
}

Legion::PrivilegeMode get_default_mode(Pass pass, TensorRole tensor_role, IsGrad is_grad) {
  if (pass == Pass::FWD) {
    assert (is_grad == IsGrad::NO);
    if (tensor_role == TensorRole::INPUT || tensor_role == TensorRole::PARAM) {
      return READ_ONLY;
    } else {
      return WRITE_ONLY;
    }
  } else {
    if (tensor_role == TensorRole::INPUT || tensor_role == TensorRole::PARAM) {
      if (is_grad == IsGrad::NO) {
        return READ_ONLY;
      } else {
        return READ_WRITE; 
      }
    } else {
      if (is_grad == IsGrad::NO) {
        return READ_ONLY;
      } else {
        return READ_WRITE; // TODO @lockshaw is this really right?
      }
    }
  }
}

void Op::execute_task_spec(FFModel const &ff, TaskSpec const &task_spec) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  this->set_argumentmap_for_forward(ff, argmap);
  TaskArgument task_arg;
  IndexLauncher launcher(task_spec.task_id, this->parallel_is, task_spec.argument, argmap, Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/, get_std_hash(this->outputs.at(0)->machine_view));

  int field_idx = 0;
  for (auto const &tensor_spec : task_spec.tensors) {
    ParallelTensor const &parallel_tensor = this->get_parallel_tensor(tensor_spec.role, tensor_spec.idx);

    if (tensor_spec.role == TensorRole::INPUT && tensor_spec.is_grad == IsGrad::YES && !this->trainableInputs.at(tensor_spec.idx)) {
      continue;
    }
    launcher.add_region_requirement(RegionRequirement(parallel_tensor->part, 
                                                      0 /*projection id*/,
                                                      tensor_spec.mode.value(),
                                                      EXCLUSIVE,
                                                      parallel_tensor->region));
    launcher.add_field(field_idx, FID_DATA);
    field_idx++;
  }

  runtime->execute_index_space(ctx, launcher);
}

}; // namespace FlexFlow
