#include "task_spec.h"
#include "utils/not_implemented_exception.h"

using namespace Legion;

namespace FlexFlow {

bool is_variadic(TensorArgumentFormat const &fmt) {
  return holds_alternative<VariadicFormat>(fmt);
}

VariadicFormat get_variadic_format(TensorArgumentFormat const &fmt) {
  return get<VariadicFormat>(fmt);
}

NonvariadicFormat get_nonvariadic_format(TensorArgumentFormat const &fmt) {
  return get<NonvariadicFormat>(fmt);
}

Legion::PrivilegeMode get_privileges(TaskArgumentFormat const &fmt,
                                     region_idx_t region_idx) {
  return fmt.regions.at(region_idx);
}

Legion::PrivilegeMode get_privileges(TaskArgumentFormat const &fmt,
                                     ParallelTensorSpec const &t) {
  return get_privileges(fmt, get_region_idx(fmt, t));
}

DataType get_datatype(TaskArgumentFormat const &fmt,
                      ParallelTensorSpec const &t) {
  return fmt.data_types.at(t);
}

region_idx_t get_region_idx(TaskArgumentFormat const &fmt,
                            ParallelTensorSpec const &t) {
  for (TensorArgumentFormat &tensor_fmt : values(fmt.region_idxs)) {
    if (is_variadic(tensor_fmt)) {
      for (auto const &f : get_variadic_format(tensor_fmt)) {
        if (t == f.second) {
          return f.first;
        }
      }
    } else {
      NonvariadicFormat f = get_nonvariadic_format(tensor_fmt);
      if (t == f.second) {
        return f.first;
      }
    }
  }
}

TaskArgumentFormat compile_task_invocation(TaskSignature const &signature,
                                           TaskBinding const &binding) {
  OpTaskArgumentFormat result;

  result.region_idxs = allocate_region_idxs(signature, binding);
  result.argument_offsets = allocate_argument_offsets(signature, binding);
  *((OpTaskArgumentFormat *)binding.task_format_location) = result;

  return result;
}

void execute_task(LegionConfig const &config,
                  TaskID task_id,
                  TaskSignature const &signature,
                  TaskBinding const &binding) {
  TaskArgumentFormat task_arg_fmt = compile_task_invocation(signature, binding);

  ArgumentMap argmap;
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;

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

    ParallelTensor const &parallel_tensor =
        this->get_parallel_tensor(tensor_spec);

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

TaskSignature get_signature(TaskID task_id) {
  switch (task_id) {
    case AGGREGATE_INIT_TASK_ID:
      return get_signature<AGGREGATE_INIT_TASK_ID>();
    case AGGREGATE_FWD_TASK_ID:
      return get_signature<AGGREGATE_FWD_TASK_ID>();
    case AGGREGATE_BWD_TASK_ID:
      return get_signature<AGGREGATE_BWD_TASK_ID>();
    case CONV2D_BWD_TASK_ID:
      return get_signature<CONV2D_BWD_TASK_ID>();
    default:
      throw mk_runtime_error(
          "Unknown task id {}. Please report this as an issue.", task_id);
  }
}

} // namespace FlexFlow
