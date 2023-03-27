#include "operator.h"
#include "op-attrs/ffconst_utils.h"
#include <stdexcept>

using namespace Legion;

namespace FlexFlow {

OpTasksSpec::OpTasksSpec(TaskID init, TaskID fwd, TaskID bwd)
  : init_spec(init), fwd_spec(fwd), bwd_spec(bwd)
{ }

OpTaskSpec &OpTasksSpec::get_init() {
  return this->init_spec;
}

OpTaskSpec &OpTasksSpec::get_fwd() {
  return this->fwd_spec;
}

OpTaskSpec &OpTasksSpec::get_bwd() {
  return this->bwd_spec;
}

OpTaskSpec const &OpTasksSpec::get_task_spec(OpTaskType task_type) const {
  switch (task_type) {
    case OpTaskType::INIT:
      return this->get_init();
    case OpTaskType::FWD:
      return this->get_fwd();
    case OpTaskType::BWD:
      return this->get_bwd();
  }
}

TensorSpec OpTasksSpec::input_tensor(int idx) {
  return TensorSpec(TensorRole::INPUT, idx);
}

TensorSpec OpTasksSpec::output_tensor(int idx) {
  return TensorSpec(TensorRole::OUTPUT, idx);
}

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

OpTaskSpec Op::infer_bwd_spec(TaskID bwd_task_id, OpTaskSpec const &fwd_spec) const {
  OpTaskSpec bwd_spec(bwd_task_id, OpTaskType::BWD);

  for (auto const &kv : fwd_spec.get_slots()) {
    int slot = kv.first;
    TensorRole role = kv.second;

    bwd_spec.add_slot(slot, role);
    bwd_spec.add_grad_slot(slot, role);

    optional<TensorSpec const &> tensor = fwd_spec.in_slot(slot);
    if (tensor.has_value()) {
      bwd_spec.bind(slot, *tensor);
      if (tensor->role == TensorRole::INPUT && !this->trainableInputs.at(tensor->idx)) {
      } else {
        bwd_spec.bind_grad(slot, tensor->grad());
      }
    }
  }

  return bwd_spec;
}

void Op::infer_bwd_spec(OpTasksSpec &spec) const {
  spec.set_bwd(this->infer_bwd_spec(spec.get_task_id(OpTaskType::BWD), spec.get_fwd()));
}

bool Op::check_output_input_weight_same_parallel_is() const {
  assert(numOutputs > 0);
  IndexSpace parallel_is = outputs[0]->parallel_is;
  for (int i = 0; i < numOutputs; i++) {
    if (outputs[i]->parallel_is != parallel_is) {
      return false;
    }
  }
  for (int i = 0; i < numInputs; i++) {
    if (inputs[i]->parallel_is != parallel_is) {
      return false;
    }
  }
  for (int i = 0; i < numWeights; i++) {
    if (weights[i]->parallel_is != parallel_is) {
      return false;
    }
  }
  return true;
}


void Op::execute_task_spec(FFModel const &ff, OpTaskSpec const &task_spec) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  this->set_argumentmap_for_forward(ff, argmap);
  TaskArgument task_arg;

  IndexLauncher launcher(task_spec.get_task_id(), this->parallel_is, task_spec.get_argument(), argmap, Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/, get_std_hash(this->outputs.at(0)->machine_view));

  for (auto const &kv : task_spec.get_region_idxs()) {
    TensorSpec const &tensor_spec = kv.first;
    int region_idx = kv.second;
    ParallelTensor const &parallel_tensor = this->get_parallel_tensor(tensor_spec.role, tensor_spec.idx);

    if (tensor_spec.role == TensorRole::INPUT && tensor_spec.is_grad == IsGrad::YES && !this->trainableInputs.at(tensor_spec.idx)) {
      continue;
    }
    launcher.add_region_requirement(RegionRequirement(parallel_tensor->part, 
                                                      0 /*projection id*/,
                                                      tensor_spec.mode.value(),
                                                      EXCLUSIVE,
                                                      parallel_tensor->region));
    launcher.add_field(region_idx, FID_DATA);
    region_idx++;
  }

  runtime->execute_index_space(ctx, launcher);
}

void Op::execute_task(FFModel const &ff, TaskID task_id, OpTaskSignature const &signature) {
  if (signature.get_task_type() == OpTaskType::INIT) {
    assert (this->check_output_input_weight_same_parallel_is());
    this->parallel_is = outputs[0]->parallel_is;
  }

  OpTaskBinding binding = this->get_task_binding(signature.get_task_type());

  OpTaskArgumentFormat task_arg_fmt = compile_task_invocation(signature, binding);

  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;

  this->set_argumentmap(signature.get_task_type(), ff, argmap);
  TaskArgument task_arg;

  IndexLauncher launcher(task_id, 
                         this->parallel_is, 
                         binding.get_legion_task_arg(), 
                         argmap, 
                         Predicate::TRUE_PRED, 
                         false /*must*/, 
                         0 /*mapper_id*/, 
                         get_std_hash(this->outputs.at(0)->machine_view));

  for (auto const &kv : get_region_idxs(task_arg_fmt)) {
    int region_idx = kv.second;
    TensorSpec const &tensor_spec = kv.second;

    ParallelTensor const &parallel_tensor = this->get_parallel_tensor(tensor_spec);

    launcher.add_region_requirement(RegionRequirement(parallel_tensor->part, 
                                                      0 /*projection id*/,
                                                      tensor_spec.mode.value(),
                                                      EXCLUSIVE,
                                                      parallel_tensor->region));
    launcher.add_field(region_idx, FID_DATA);
    region_idx++;
  }

  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  if (signature.get_task_type() == OpTaskType::INIT) {
    fm.wait_all_results();
    this->set_opmeta_from_futuremap(ff, fm);
  }
}

void Op::init(FFModel const &ff) {
  this->execute_task_spec(ff, this->get_fully_defined_tasks_spec().get_init());
}

void Op::forward(FFModel const &ff) {
  this->execute_task_spec(ff, this->get_fully_defined_tasks_spec().get_fwd());
}

void Op::backward(FFModel const &ff) {
  this->execute_task_spec(ff, this->get_fully_defined_tasks_spec().get_bwd());
}

}
