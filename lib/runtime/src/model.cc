/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "model.h"
/* #if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA) */
/* #include "flexflow/utils/cuda_helper.h" */
/* #else */
/* #include "utils/hip_helper.h" */
/* #endif */
#include "legion_parallel_tensor_shape.h"
#include "mapper.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task_argument_accessor.h"
#include "utils/random_utils.h"
#include "test_utils.h"
#include "legion/legion_utilities.h"
#include <dirent.h>
#include <queue>
#include <unordered_set>
#include "utils/containers.h"
#include "parallel_tensor_mapping.h"
#include "op-attrs/ops/noop.h"

using namespace Legion;

namespace FlexFlow {


/* std::unordered_map<int, int> output_to_input_mapping( */
/*     std::vector<ParallelDimMappingRecord> const &mapping) { */
/*   std::unordered_map<int, int> dim_mapping; */
/*   for (ParallelDimMappingRecord const &record : mapping) { */
/*     if (record.get_type() == MappingRecordType::INPUT_OUTPUT) { */
/*       dim_mapping[record.output_dim] = record.input_dim; */
/*     } */
/*   } */

/*   return dim_mapping; */
/* } */

/* std::unordered_map<int, int> input_to_output_mapping( */
/*     std::vector<ParallelDimMappingRecord> const &mapping) { */
/*   std::unordered_map<int, int> dim_mapping; */
/*   for (ParallelDimMappingRecord const &record : mapping) { */
/*     if (record.get_type() == MappingRecordType::INPUT_OUTPUT) { */
/*       dim_mapping[record.input_dim] = record.output_dim; */
/*     } */
/*   } */

/*   return dim_mapping; */
/* } */

FFModel::FFModel(FFConfig const &_config, 
                 ComputationGraph const &cg, 
                 ParallelComputationGraph const &pcg, 
                 Optimizer const &_optimizer)
    : config(_config), 
      index_space_mgr(_config.legion_config), 
      computation_graph(cg),
      pcg(pcg),
      optimizer(_optimizer) {

  Runtime *runtime = config.legion_config.lg_hlr;
  Context ctx = config.legion_config.lg_ctx;
  // Register machine views
  register_all_machine_views(config.numNodes,
                             config.workersPerNode,
                             config.cpusPerNode,
                             all_valid_views);
  metrics_input = -1;
  // Create field space
  {
    FieldAllocator allocator =
        runtime->create_field_allocator(ctx, config.legion_config.field_space);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }

  ArgumentMap argmap;
  Rect<1> task_rect(Point<1>(0),
                    Point<1>(config.workersPerNode * config.numNodes - 1));
  IndexSpaceT<1> task_is = runtime->create_index_space(ctx, task_rect);

  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    FFInitInfo info;
    info.workSpaceSize = config.workSpaceSize;
    info.allowTensorOpMathConversion = config.allow_tensor_op_math_conversion;
    argmap.set_point(*it, TaskArgument(&info, sizeof(FFInitInfo)));
  }

  // Init CUDA library on each worker
  IndexLauncher initLauncher(FF_INIT_TASK_ID,
                             task_is,
                             TaskArgument(NULL, 0),
                             argmap,
                             Predicate::TRUE_PRED,
                             false /*must*/,
                             0 /*mapper_id*/,
                             FFConfig::DataParallelism_GPU);
  FutureMap fm = runtime->execute_index_space(ctx, initLauncher);
  fm.wait_all_results();
  int idx = 0;
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    handlers[idx++] = fm.get_result<FFHandler>(*it);
  }
}

using FullyExecutableArgSpec = variant<ConcreteArgSpec, CheckedTypedFuture>;

struct ArgumentsConstructionState {
  variant<Legion::TaskLauncher, Legion::IndexLauncher> launcher;
  int num_futures;
  Legion::Serializer sez;
  TaskArgumentsFormat args_fmt;
};

struct AddArgumentToTaskFunctor {
  AddArgumentToTaskFunctor(ArgumentsConstructionState &state, slot_id slot) : state(state), slot(slot) { }

  ArgumentsConstructionState &state;
  slot_id slot;

  void operator()(ConcreteArgSpec const &a) { 
    size_t start = state.sez.get_used_bytes();
    a.serialize(state.sez);
    size_t end = state.sez.get_used_bytes();
    state.args_fmt.insert({slot, TaskArgumentFormat(a.get_type_tag().get_type_idx(), start, end)});
  }

  void operator()(CheckedTypedFuture const &a) { 
    if (holds_alternative<Legion::TaskLauncher>(state.launcher)) {
      get<Legion::TaskLauncher>(state.launcher).add_future(a.get_unsafe());
    } else {
      get<Legion::IndexLauncher>(state.launcher).add_future(a.get_unsafe());
    }
    state.args_fmt.insert({slot, FutureArgumentFormat(a.get_type_tag().get_type_idx(), state.num_futures)});
    state.num_futures++;
  }
};

static TaskArgumentFormat add_argument_to_task_arg(ArgumentsConstructionState &state,
                                                   slot_id slot, 
                                                   FullyExecutableArgSpec const &arg_spec) {
  visit(AddArgumentToTaskFunctor{state, slot}, arg_spec);
}

static ConcreteArgSpec resolve_index_arg(IndexArgSpec const &index_arg, Legion::DomainPoint const &);
static CheckedTypedFuture resolve_future_map_arg(CheckedTypedFutureMap const &future_map, Legion::DomainPoint const &);

struct ConcreteArgsFormat {
  ConcreteArgsFormat() = delete;
  ConcreteArgsFormat(Legion::Serializer const &sez, TaskArgumentsFormat *reserved_bytes_for_fmt, stack_map<slot_id, TaskArgumentFormat, MAX_NUM_TASK_ARGUMENTS> const &fmts)
    : sez(sez), fmts(fmts)
  { }

  Legion::Serializer sez;
  TaskArgumentsFormat *reserved_bytes_for_fmt;
  stack_map<slot_id, TaskArgumentFormat, MAX_NUM_TASK_ARGUMENTS> fmts;
};

template <typename T>
std::unordered_map<slot_id, T> get_args_of_type(ExecutableTaskBinding const &binding) {
  static_assert(is_in_variant<T, ExecutableArgSpec>::value, "");
  return map_values(filter_values(binding.arg_bindings, 
                                  [](ExecutableArgSpec const &s) { return holds_alternative<T>(s); }),
                    [](ExecutableArgSpec const &s) { return get<T>(s); });
            
}

ConcreteArgsFormat process_concrete_args(std::unordered_map<slot_id, ConcreteArgSpec> const &specs) {
  Legion::Serializer sez;
  TaskArgumentsFormat *reserved = static_cast<TaskArgumentsFormat*>(sez.reserve_bytes(sizeof(TaskArgumentsFormat)));
  stack_map<slot_id, TaskArgumentFormat, MAX_NUM_TASK_ARGUMENTS> fmts;
  for (auto const &kv : specs) {
    slot_id slot = kv.first;
    ConcreteArgSpec arg = kv.second;

    size_t before = sez.get_used_bytes();
    arg.serialize(sez);
    size_t after = sez.get_used_bytes();

    fmts.insert(slot, {arg.get_type_tag().get_type_idx(), before, after});
  }
  return { sez, reserved, fmts };

}

ConcreteArgsFormat process_concrete_args(ExecutableTaskBinding const &binding) {
  return process_concrete_args(get_args_of_type<ConcreteArgSpec>(binding));
}

struct FutureArgsFormat {
  std::vector<Future> futures;
  stack_map<slot_id, FutureArgumentFormat, MAX_NUM_TASK_ARGUMENTS> fmts;
};

FutureArgsFormat process_future_args(ExecutableTaskBinding const &binding) {
  std::vector<Future> futures;
  stack_map<slot_id, FutureArgumentFormat, MAX_NUM_TASK_ARGUMENTS> fmts;

  for (auto const &kv : get_args_of_type<CheckedTypedFuture>(binding)) {
    slot_id slot = kv.first;
    CheckedTypedFuture fut = kv.second;

    futures.push_back(fut.get_unsafe());
    FutureArgumentFormat fmt = {fut.get_type_idx(), futures.size()-1} ;
    fmts.insert(slot, fmt);
  }
  return { futures, fmts };
};

struct IndexArgsFormat : public use_visitable_cmp<IndexArgsFormat> {
  IndexArgsFormat() = delete;
  IndexArgsFormat(std::map<Legion::DomainPoint, ConcreteArgsFormat> const &point_map)
    : point_map(point_map)
  { }

public:
  std::map<Legion::DomainPoint, ConcreteArgsFormat> point_map;
};

ConcreteArgsFormat process_index_args_for_point(std::unordered_map<slot_id, IndexArgSpec> const &specs,
                                                Legion::DomainPoint const &p) {
  std::unordered_map<slot_id, ConcreteArgSpec> resolved = map_values(specs,
              [&](IndexArgSpec const &s) { return resolve_index_arg(s, p); });
  return process_concrete_args(resolved);
}

void add_futures(TaskLauncher const &, FutureArgsFormat const &);
void add_futures(IndexTaskLauncher const &, FutureArgsFormat const &);

IndexArgsFormat process_index_args(ExecutableTaskBinding const &binding, Legion::Domain const &domain) {
  std::map<Legion::DomainPoint, ConcreteArgsFormat> point_map;
  auto index_args = get_args_of_type<IndexArgSpec>(binding);
  for (Legion::Domain::DomainPointIterator it(domain); it; it++) {
    point_map.insert({*it, process_index_args_for_point(index_args, *it)}); 
  }
  return { point_map };
};

struct TensorArgsFormat {
  bidict<parallel_tensor_guid_t, region_idx_t> region_idxs;
  std::unordered_map<parallel_tensor_guid_t, Permissions> privs_map;
  std::unordered_map<parallel_tensor_guid_t, DataType> datatypes;
  std::unordered_map<slot_id, parallel_tensor_guid_t> nonvariadic_slot_to_tensor;
  std::unordered_map<slot_id, std::vector<parallel_tensor_guid_t>> variadic_slot_to_tensor;
};

void add_tensor_requirements(TaskLauncher const &, TensorArgsFormat const &);
void add_tensor_requirements(IndexTaskLauncher const &, TensorArgsFormat const &);

static bool includes_tensor(ExecutableTensorSpec const &spec, parallel_tensor_guid_t guid) {
  if (is_variadic(spec)) {
    return contains(get_variadic(spec), guid);
  } else {
    assert (is_nonvariadic(spec));
    return get_nonvariadic(spec) == guid;
  }
}

std::unordered_set<slot_id> get_tensor_slots(ExecutableTaskBinding const &binding, parallel_tensor_guid_t guid) {
  std::unordered_set<slot_id> results; 
  for (auto const &kv : binding.tensor_bindings) {
    slot_id slot = kv.first;
    ExecutableTensorSpec spec = kv.second;
    if (includes_tensor(spec, guid)) {
      results.insert(slot);
    }
  }
  return results;
}

Permissions get_tensor_permissions(TaskSignature const &sig, ExecutableTaskBinding const &binding, parallel_tensor_guid_t guid) {
  Permissions result = Permissions::NONE;
  for (slot_id slot : get_tensor_slots(binding, guid)) {
    result = join(result, sig.get_slot(slot)->perm);
  }
  return result;
}

std::vector<parallel_tensor_guid_t> as_vector(ExecutableTensorSpec const &spec) {
  if (is_variadic(spec)) {
    return get_variadic(spec);
  } else {
    assert (is_nonvariadic(spec));
    return { get_nonvariadic(spec) };
  }
}

TensorArgsFormat process_tensor_args(TaskSignature const &sig, 
                                     ParallelComputationGraph const &pcg,
                                     ExecutableTaskBinding const &binding) {
  std::unordered_map<parallel_tensor_guid_t, Permissions> privs_map;
  bidict<parallel_tensor_guid_t, region_idx_t> region_idxs;
  std::unordered_map<parallel_tensor_guid_t, DataType> datatypes;
  int idx_ctr = 0;
  for (parallel_tensor_guid_t const &guid : unique(flatmap(values(binding.tensor_bindings), as_vector))) {
    for (slot_id slot : get_tensor_slots(binding, guid)) {
      privs_map[guid] = get_tensor_permissions(sig, binding, guid);
      region_idx_t idx = region_idx_t(idx_ctr++);
      region_idxs.equate(guid, idx);
      datatypes[guid] = pcg.at(guid).data_type;
    }
  }
  std::unordered_map<slot_id, parallel_tensor_guid_t> nonvariadic_slot_to_tensor;
  std::unordered_map<slot_id, std::vector<parallel_tensor_guid_t>> variadic_slot_to_tensor;
  for (slot_id slot : keys(binding.tensor_bindings)) {
    if (is_variadic(binding, slot)) {
      variadic_slot_to_tensor[slot] = get_variadic(binding.tensor_bindings.at(slot));
    } else {
      assert (is_nonvariadic(binding, slot));
      nonvariadic_slot_to_tensor[slot] = get_nonvariadic(binding.tensor_bindings.at(slot));
    }
  }

  return { region_idxs, privs_map, datatypes, nonvariadic_slot_to_tensor, variadic_slot_to_tensor };
}

TaskArgumentsFormat create_serializable_format(TensorArgsFormat const &tensor_args_format,
                                               ConcreteArgsFormat const &concrete_args_format,
                                               FutureArgsFormat const &future_args_format,
                                               optional<IndexArgsFormat> const &index_args_format = nullopt) {
  TaskArgumentsFormat result;
  for (auto const &kv : concrete_args_format.fmts) {
    result.insert(kv);
  }
  for (auto const &kv : future_args_format.fmts) {
    result.insert(kv);
  }
  assert (!index_args_format.has_value());
  for (parallel_tensor_guid_t const &guid : keys(tensor_args_format.region_idxs)) {
    region_idx_t region_idx = tensor_args_format.region_idxs.at_l(guid);
    Legion::PrivilegeMode privs = to_legion(tensor_args_format.privs_map.at(guid));
    DataType datatype = tensor_args_format.datatypes.at(guid);
    result.insert(region_idx, privs, datatype);
  }
  for (auto const &kv : tensor_args_format.nonvariadic_slot_to_tensor) {
    slot_id slot = kv.first;
    parallel_tensor_guid_t guid = kv.second;
    region_idx_t region_idx = tensor_args_format.region_idxs.at_l(guid);
    result.insert(slot, region_idx);
  }
  for (auto const &kv : tensor_args_format.variadic_slot_to_tensor) {
    slot_id slot = kv.first;
    std::vector<parallel_tensor_guid_t> guids = kv.second;
    std::vector<region_idx_t> region_idxs = transform(guids, lookup_in_l(tensor_args_format.region_idxs));
    result.insert(slot, region_idxs);
  }
  return result;
}


std::vector<ExecutableArgSpec> process_task_invocation_args(FFModel const &model, ExecutableTaskBinding const &binding) {
  for (auto const &kv : get_args_of_type<TaskInvocationSpec>(binding)) {
    slot_id slot = kv.first;
    TaskInvocationSpec spec = kv.second;
    ExecutableTaskInvocation executable = model.resolve(spec.get_invocation());

    TaskReturnAccessor ret_val = model.execute(executable);
    
  }
}

Legion::TaskArgument as_task_argument(ConcreteArgsFormat const &concrete_args_format, 
                                      FutureArgsFormat const &future_args_format, 
                                      TensorArgsFormat const &tensor_args_format, 
                                      optional<IndexArgsFormat const &> index_args_format = nullopt) {
  TaskArgumentsFormat serializable_format = create_serializable_format(tensor_args_format,
                                                                       concrete_args_format,
                                                                       future_args_format,
                                                                       index_args_format);
  *(concrete_args_format.reserved_bytes_for_fmt) = serializable_format;
  return Legion::TaskArgument(concrete_args_format.sez.get_buffer(),
                              concrete_args_format.sez.get_used_bytes());
}
Legion::ArgumentMap as_argument_map(IndexArgsFormat const &);

TaskReturnAccessor FFModel::execute(ExecutableTaskInvocation const &invocation) const {
  TaskSignature sig = get_signature(invocation.task_id);
  ExecutableTaskBinding binding = invocation.binding;
  TensorArgsFormat tensor_args_format = process_tensor_args(sig, this->pcg, binding);
  ConcreteArgsFormat concrete_args_format = process_concrete_args(binding);
  FutureArgsFormat future_args_format = process_future_args(binding);
  TaskInvocationArgsFormat task_invocation_args_format = process_task_invocation_args(*this, binding);
  assert (get_args_of_type<CheckedTypedFutureMap>(binding).empty()); // currently we don't handle these as I don't think they're used anywhere
  if (binding.invocation_type == InvocationType::STANDARD) {
    assert (get_args_of_type<IndexArgSpec>(binding).empty());
    Legion::TaskArgument task_arg = as_task_argument(concrete_args_format,
                                                     future_args_format,
                                                     tensor_args_format);
    TaskLauncher launcher(invocation.task_id, task_arg);
    add_tensor_requirements(launcher, tensor_args_format);
    Future returned_future = this->runtime_backing.execute_task(launcher);
    return TaskReturnAccessor(sig.get_return_type(), returned_future);
  } else if (binding.invocation_type == InvocationType::INDEX) {
    parallel_tensor_guid_t index_space_determiner = binding.domain_spec.value();
    ParallelTensorBacking backing = this->runtime_backing.at(index_space_determiner);
    IndexArgsFormat index_args_format = process_index_args(binding, 
                                                           this->runtime_backing.get_domain(backing.parallel_is));
    Legion::TaskArgument task_arg = as_task_argument(concrete_args_format,
                                                     future_args_format,
                                                     tensor_args_format,
                                                     index_args_format);
    IndexTaskLauncher launcher(invocation.task_id,
                               backing.parallel_is,
                               task_arg,
                               as_argument_map(index_args_format),
                               Predicate::TRUE_PRED,
                               false /*must*/,
                               0 /*mapper_id*/,
                               backing.mapping_id.value()
                               );
    add_tensor_requirements(launcher, tensor_args_format);
    FutureMap returned_future = this->runtime_backing.execute_task(launcher);
    return TaskReturnAccessor(sig.get_return_type(), returned_future);
  }
}

void FFModel::init_operators() {
  for (auto const &op : operators) {
    op->init(*this);
  }
}

void FFModel::forward(int seq_length) {
  iter_config.seq_length = seq_length;
  forward(this->pcg);
  for (auto const &op : operators) {
    op->forward(*this);
  }
}

void FFModel::recompile_on_condition(RecompileState &r) {
  if (r.trigger()) {
    r.alter();
  }
}

void FFModel::compute_metrics() {
  Operator final_operator = get_final_operator();
  assert(final_operator->numOutputs == 1);
  metrics_op->compute(this, final_operator->outputs[0], parallel_label_tensor.value());
}

void FFModel::backward(int seq_length) {
  iter_config.seq_length = seq_length;
  assert(config.computationMode == COMP_MODE_TRAINING);
  // Compute metrics
  compute_metrics();
  // Compute the gradients of the final operator wrt loss
  Op const *final_operator = get_final_operator();
  assert(final_operator->numOutputs == 1);
  loss_op->backward(this, final_operator->outputs[0], parallel_label_tensor.value());
  // Perform backpropagation
  // std::set<LogicalRegion> resetedInputGrads;
  for (int l = operators.size() - 1; l >= 0; l--) {
#ifdef ENABLE_RESNET_INPUT_GRADIENT_OPTIMIZATION
    for (int i = 0; i < operators[l]->numInputs; i++) {
      if (resetedInputGrads.find(operators[l]->inputs[i]->region) ==
          resetedInputGrads.end()) {
        resetedInputGrads.insert(operators[l]->inputs[i]->region);
      } else {
        // This input's gradients has been reseted by other operators
        // So we should not do it again
        operators[l]->resetInputGrads[i] = false;
      }
    }
#endif
    // TODO: If operator serves for metrics and for further prop
    // if(l == metrics_input && metrics_input < (int)operators.size()-1)
    //  continue;
    operators[l]->backward(*this);
  }
}

void FFModel::update() {
  optimizer->next();
  for (size_t i = 0; i < parameters.size(); i++) {
    optimizer->update(parameters[i]);
  }
}

Op const *FFModel::get_final_operator() const {
  int idx = operators.size() - 1;
  std::vector<Op const *> operators = this->get_operators();
  while (operators[idx]->op_type == OP_INPUT ||
         operators[idx]->op_type == OP_WEIGHT) {
    idx--;
  }
  // assert that the final operator has exactly one output
  assert(operators[idx]->numOutputs == 1);
  return operators.at(idx);
}

void FFModel::compile(Optimizer *_optimizer,
                      LossType loss_type,
                      std::vector<MetricsType> const &metrics,
                      CompMode comp_mode) {
  optimizer = _optimizer;
  compile(loss_type, metrics, comp_mode);
}

bool FFModel::apply_fusion(std::vector<Op *> const &operators,
                           std::vector<Op *> &new_operators) {
  // Context ctx = config.lg_ctx;
  // Runtime* runtime = config.lg_hlr;
  for (size_t l = 1; l < operators.size() - 1; l++) {
    // don't fuse input and weight operator since they don't involve any
    // forward/backward task launches
    if (operators[l]->op_type == OP_INPUT ||
        operators[l]->op_type == OP_WEIGHT) {
      continue;
    }
    // don't fuse parallel op since they have different parallel_is in
    // forward/backward
    if (operators[l]->is_parallel_op()) {
      continue;
    }
    size_t start = 0;
    {
      Op *opl = operators[l];
      for (int idx = 0; idx < opl->numInputs; idx++) {
        bool found = false;
        for (size_t i = 0; i < l; i++) {
          if (opl->inputs[idx]->owner_op == operators[i]) {
            assert(!found);
            found = true;
            if (i > start) {
              start = i;
            }
          }
        }
        assert(found || (opl->inputs[idx]->owner_op == NULL));
      }
    }
    for (size_t i = start; i < l; i++) {
      // Domain d1 =
      // runtime->get_index_space_domain(operators[l]->outputs[0]->parallel_is);
      // Domain d2 =
      // runtime->get_index_space_domain(operators[i]->outputs[0]->parallel_is);
      MachineView view1 = operators[l]->outputs[0]->machine_view.value();
      MachineView view2 = operators[i]->outputs[0]->machine_view.value();
      if (view1 == view2) {
        FusedOp *fused_op = nullptr;
        bool allocate_new_fused_op = false;
        if (operators[i]->op_type == OP_FUSED) {
          fused_op = (FusedOp *)operators[i];
        } else {
          //  cannot be an in-place operator
          if (operators[i]->has_inplace_output()) {
            continue;
          }
          // don't fuse input and weight operator since they don't involve any
          // forward/backward kernels
          if (operators[i]->op_type == OP_INPUT ||
              operators[i]->op_type == OP_WEIGHT) {
            continue;
          }
          // don't fuse parallel op since they have different parallel_is in
          // forward/backward
          if (operators[i]->is_parallel_op()) {
            continue;
          }
          fused_op = new FusedOp(*this, operators[i]);
          allocate_new_fused_op = true;
        }
        if (fused_op->add_operator(*this, operators[l])) {
          // Construct new operators
          new_operators.clear();
          for (size_t j = 0; j < i; j++) {
            new_operators.push_back(operators[j]);
          }
          new_operators.push_back(fused_op);
          for (size_t j = i + 1; j < operators.size(); j++) {
            if (j == l) {
              continue; // l and i are fused
            }
            Op *op = operators[j];
            // Update input tensors that belong to operator[l] or operator[i]
            for (int idx = 0; idx < op->numInputs; idx++) {
              if ((op->inputs[idx]->owner_op == operators[l]) ||
                  (op->inputs[idx]->owner_op == operators[i])) {
                int found = -1;
                for (int k = 0; k < fused_op->numOutputs; k++) {
                  if (fused_op->outputs[k]->region == op->inputs[idx]->region) {
                    assert(found == -1);
                    found = k;
                  }
                }
                assert(found >= 0);
                op->inputs[idx] = fused_op->outputs[found];
              }
            }
            // Insert op
            new_operators.push_back(op);
          }
          // We are exact one operator fewer than the original
          assert(new_operators.size() + 1 == operators.size());
          return true;
        } else {
          // TODO: delete fused_op to avoid memory leakage
          if (allocate_new_fused_op) {
            delete fused_op;
          }
          continue;
        }
      }
    }
  }
  return false;
}

static ParallelTensorShape get_parallel_tensor_shape(Tensor const &tensor) {
  int num_dims = tensor->num_dims();
  std::vector<ParallelDim> dims;
  for (int j = 0; j < num_dims; j++) {
    dims.emplace_back(tensor->dims[j], 1, -1, false);
  }
  dims.emplace_back(1, 1, -1, true);
  ParallelTensorShape shape = { dims, tensor->data_type };
  return shape;
}

Op *FFModel::create_operator_from_layer(
    Layer *layer, std::vector<ParallelTensor> const &inputs) {
  return make_operator_unsafe(*this, layer->attrs, inputs);

  //switch (layer->op_type) {
  //  case OP_INPUT: {
  //    // Input op cannot have an input
  //    assert(inputs.size() == 0);
  //    Tensor tensor = layer->outputs[0];
  //    // Current assume we add one dimension before each tensor
  //    // create_parallel_tensor adds an NoOp into operators
  //    ParallelTensorShape shape = get_parallel_tensor_shape(tensor);
  //    ParallelTensor pt =
  //        create_parallel_tensor(shape,
  //                                               nullptr,
  //                                               0,
  //                                               true /*gradients*/,
  //                                               tensor->tensor_guid);
  //    // assert that this tensor hasn't been mapped before
  //    assert(tensor->parallel_tensor == nullopt);
  //    tensor->parallel_tensor = pt;
  //    // start from data parllel tensor
  //    if (config.only_data_parallel) {
  //      Repartition *part = new Repartition(
  //          *this, pt, shape.num_dims() - 1, config.numNodes * config.workersPerNode);
  //      operators.push_back(part);
  //    }
  //    return operators[operators.size() - 1];
  //  }
  //  case OP_MULTIHEAD_ATTENTION: {
  //    Op *op =
  //        MultiHeadAttention::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_BATCHMATMUL: {
  //    Op *op = BatchMatmul::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_CAST: {
  //    Op *op = Cast::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_CONCAT: {
  //    Op *op = Concat::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_CONV2D: {
  //    Op *op = Conv2D::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_DROPOUT: {
  //    Op *op = Dropout::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_EMBEDDING: {
  //    Op *op = Embedding::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_EW_ADD:
  //  case OP_EW_SUB:
  //  case OP_EW_MUL:
  //  case OP_EW_DIV:
  //  case OP_EW_MAX:
  //  case OP_EW_MIN: {
  //    Op *op = ElementBinary::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_EXP:
  //  case OP_SIN:
  //  case OP_COS:
  //  case OP_SCALAR_MULTIPLY:
  //  case OP_SCALAR_ADD:
  //  case OP_SCALAR_SUB:
  //  case OP_SCALAR_TRUE_DIV:
  //  case OP_POW:
  //  case OP_RELU:
  //  case OP_SIGMOID:
  //  case OP_TANH:
  //  case OP_IDENTITY:
  //  case OP_GELU:
  //  case OP_ELU: {
  //    Op *op = ElementUnary::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_FLAT: {
  //    Op *op = Flat::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_GATHER: {
  //    Op *op = Gather::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_LAYERNORM: {
  //    Op *op = LayerNorm::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_LINEAR: {
  //    Op *op = Linear::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_POOL2D: {
  //    Op *op = Pool2D::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_REDUCE_SUM: {
  //    Op *op = Reduce::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_RESHAPE: {
  //    Op *op = Reshape::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_SOFTMAX: {
  //    Op *op = Softmax::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_SPLIT: {
  //    Op *op = Split::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_TRANSPOSE: {
  //    Op *op = Transpose::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_TOPK: {
  //    Op *op = TopK::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_GROUP_BY: {
  //    Op *op = Group_by::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_AGGREGATE: {
  //    Op *op = Aggregate::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_AGG_SPEC: {
  //    Op *op = Aggregate::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  default:
  //    assert(false);
  //}
}

void FFModel::create_operators_from_layers() {
  std::map<Tensor const, ParallelTensor> tensors_to_parallel_tensors;
  for (auto const &l : layers) {
    std::vector<ParallelTensor> inputs;
    for (int i = 0; i < l->numInputs; i++) {
      // create new input tensors
      assert(tensors_to_parallel_tensors.find(l->inputs[i]) !=
             tensors_to_parallel_tensors.end());
      inputs.push_back(tensors_to_parallel_tensors[l->inputs[i]]);
    }
    Op *op = create_operator_from_layer(l, inputs);
    assert(op->numOutputs == l->numOutputs);
    for (int i = 0; i < op->numOutputs; i++) {
      tensors_to_parallel_tensors[l->outputs[i]] = op->outputs[i];
    }
  }
}

void FFModel::perform_inplace_optimizations() {
  for (size_t l = 1; l < operators.size(); l++) {
    if (operators[l]->can_inplace_output()) {
      // Assume outputs[0] is inplace with inputs[0]
      assert(operators[l]->numOutputs == 1);
      if (operators[l]->inputs[0]->owner_op != NULL) {
        // int dim1 = operators[l]->outputs[0]->num_dims;
        // int dim2 = operators[l]->inputs[0]->num_dims;
        MachineView view1 = operators[l]->outputs[0]->machine_view.value();
        MachineView view2 = operators[l]->inputs[0]->machine_view.value();
        if (view1 == view2) {
          // Check no others also need operators[l]->inputs[0]
          bool found = false;
          for (size_t i = 0; i < operators.size(); i++) {
            if (i == l) {
              continue;
            }
            for (int j = 0; j < operators[i]->numInputs; j++) {
              if ((operators[i]->inputs[j]->owner_op ==
                   operators[l]->inputs[0]->owner_op) &&
                  (operators[i]->inputs[j]->owner_idx ==
                   operators[l]->inputs[0]->owner_idx)) {
                found = true;
              }
            }
          }
          if (!found) {
            // Perform inplace
            operators[l]->do_inplace_output();
          }
        }
      }
    }
  }
}

void FFModel::perform_fusion_optimizations() {
  fprintf(stderr, "Applying fusion optimizations during compilation...\n");
  fprintf(stderr, "%zu operators before fusion...\n", operators.size());
  std::vector<Op *> new_operators;
  std::vector<Op *> old_operators = operators;
  while (apply_fusion(operators, new_operators)) {
    for (size_t i = 0; i < new_operators.size(); i++) {
      for (int idx = 0; idx < new_operators[i]->numInputs; idx++) {
        for (size_t j = i + 1; j < new_operators.size(); j++) {
          if (new_operators[i]->inputs[idx]->owner_op == new_operators[j]) {
            assert(false);
          }
        }
      }
    }
    operators = new_operators;
  }
  // Check integrity
  for (size_t l = 0; l < operators.size(); l++) {
    if (operators[l]->op_type == OP_FUSED) {
      FusedOp *fused = (FusedOp *)operators[l];
      int ioff = 0, woff = 0, ooff = 0;
      for (int op = 0; op < fused->numOperators; op++) {
        Op *old_op = fused->operators[op];
        for (int i = 0; i < fused->op_num_inputs[op]; i++) {
          int my_off = fused->op_input_idx[i + ioff];
          if (fused->op_input_source[i + ioff] == FusedOp::SOURCE_INPUT) {
            assert(fused->inputs[my_off]->region ==
                   old_op->inputs[i]->region);
          } else if (fused->op_input_source[i + ioff] ==
                     FusedOp::SOURCE_OUTPUT) {
            assert(fused->outputs[my_off]->region ==
                   old_op->inputs[i]->region);
          } else {
            assert(false);
          }
        }
        for (int i = 0; i < fused->op_num_weights[op]; i++) {
          int my_off = fused->op_weight_idx[i + woff];
          assert(fused->op_weight_source[i + woff] == FusedOp::SOURCE_WEIGHT);
          assert(fused->weights[my_off]->region ==
                 old_op->weights[i]->region);
        }
        for (int i = 0; i < fused->op_num_outputs[op]; i++) {
          int my_off = fused->op_output_idx[i + ooff];
          assert(fused->op_output_source[i + ooff] == FusedOp::SOURCE_OUTPUT);
          assert(fused->outputs[my_off]->region ==
                 old_op->outputs[i]->region);
        }
        ioff += fused->op_num_inputs[op];
        woff += fused->op_num_weights[op];
        ooff += fused->op_num_outputs[op];
      }
    } else {
      bool found = false;
      for (size_t i = 0; i < old_operators.size(); i++) {
        if (old_operators[i] == operators[l]) {
          assert(!found);
          found = true;
        }
      }
      assert(found);
    }
  }
  fprintf(stderr, "%zu operators after fusion...\n", operators.size());
  for (size_t i = 0; i < operators.size(); i++) {
    Op *op = operators[i];
    printf("operator[%zu]: type(%s) guid(%lu)\n",
           i,
           get_operator_type_name(operators[i]->op_type).c_str(),
           operators[i]->op_guid);
    for (int j = 0; j < op->numInputs; j++) {
      LogicalRegion handle = op->inputs[j]->region;
      printf("inputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
    for (int j = 0; j < op->numOutputs; j++) {
      LogicalRegion handle = op->outputs[j]->region;
      printf("outputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
    for (int j = 0; j < op->numWeights; j++) {
      LogicalRegion handle = op->weights[j]->region;
      printf("weights[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
  }
}


void FFModel::print_operator_regions() const {
  for (size_t i = 0; i < operators.size(); i++) {
    Op *op = operators[i];
    printf("operator[%zu]: type(%d)\n", i, operators[i]->op_type);
    for (int j = 0; j < op->numInputs; j++) {
      LogicalRegion handle = op->inputs[j]->region;
      printf("inputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
    for (int j = 0; j < op->numOutputs; j++) {
      LogicalRegion handle = op->outputs[j]->region;
      printf("outputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
  }
}

void FFModel::create_label_tensor(LossType loss_type) {
  Op const *final_operator = get_final_operator();

  std::vector<ParallelDim> p_dims = final_operator->outputs[0]->get_shape().dims;

  std::vector<size_t> dims;
  // FIXME: Currently assume 1st input for 1st operator = batch_size
  for (ParallelDim const &dim : p_dims) {
    if (!dim.is_replica_dim) {
      dims.push_back(dim.size);
    }
  }

  DataType label_type = DT_FLOAT;
  if (loss_type == LOSS_SPARSE_CATEGORICAL_CROSSENTROPY) {
    // assign dims[num_dims-1] = 1 for sparse categorical labels
    assert(p_dims[0].degree == 1);
    p_dims[0].size = 1;
    dims[0] = 1;
    label_type = DT_INT32;
  }

  LegionParallelTensorShape label_p_shape = { p_dims, label_type };
  LegionTensorShape label_shape = { dims, label_type };

  // create label tensor
  label_tensor = create_tensor(label_shape, NULL, 0 /*idx*/, false /*create_grad*/);   
  parallel_label_tensor = create_parallel_tensor(label_p_shape);                                       
  label_tensor.value()->parallel_tensor = parallel_label_tensor;                     
  parallel_label_tensor.value()->machine_view =                                      
      final_operator->outputs[0]->machine_view;                              
  map_tensor(parallel_label_tensor.value(), 
             parallel_label_tensor.value()->owner_op,
             this->config.legion_config,
             this->index_space_mgr);
}

void FFModel::execute_graph_optimize() {
  FFModel *model = this;
  Context ctx = config.legion_config.lg_ctx;
  Runtime *runtime = config.legion_config.lg_hlr;
  TaskLauncher launcher(GRAPH_OPTIMIZE_TASK_ID,
                        TaskArgument(&model, sizeof(FFModel *)));
  Future future = runtime->execute_task(ctx, launcher);

  PCG::GraphOptimalViewSerialized ret =
      future.get_result<PCG::GraphOptimalViewSerialized>();
  Deserializer dez(ret.data, ret.total_bytes);
  // Reconstruct operators
  PCG::Graph *best_graph = new PCG::Graph(this);
  std::unordered_map<PCG::Node, MachineView> optimal_views;
  deserialize_graph_optimal_view(dez, best_graph, optimal_views);
  operators.clear();
  convert_graph_to_operators(best_graph, optimal_views);
  best_graph->print_dot();
  delete best_graph;

  this->populate_tensor_to_parallel_tensor_mapping();
}

void FFModel::compile(LossType loss_type,
                      std::vector<MetricsType> const &metrics,
                      CompMode comp_mode) {
  if (metrics_input == -1) {
    metrics_input = operators.size() - 1;
  }
  Context ctx = config.legion_config.lg_ctx;
  Runtime *runtime = config.legion_config.lg_hlr;
  config.computationMode = comp_mode;
  // if (config.import_strategy_file.length() > 0) {
  //   load_strategies_from_file(config.import_strategy_file,
  //   config.strategies);
  // }
  //  Construct operators from layers
  if (config.only_data_parallel) {
    fprintf(stderr,
            "Note: only_data_parallel is specified, FlexFlow compiles a "
            "data-parallel PCG.\n");
  }
  this->create_operators_from_layers();
  
  // Launch the graph optimize task
  this->execute_graph_optimize();

  bool repl_labels = (operators[operators.size() - 1]->op_type == OP_AGG_SPEC);
  loss_op = {loss_type, repl_labels};
  metrics_op = {loss_type, metrics};

  // Init performance metrics
  TaskLauncher launcher(UPDATE_METRICS_TASK_ID,
                        TaskArgument(&metrics_op.value(), sizeof(Metrics)));
  current_metrics = runtime->execute_task(ctx, launcher);

  if (config.enable_inplace_optimizations) {
    this->perform_inplace_optimizations();
  }

  for (Op *op : this->operators) {
    for (ParallelTensor const &input : op->inputs) {
      assert(input->owner_op != NULL);
    }

    for(ParallelTensor const &weight : op->weights) {
      assert(weight->owner_op != NULL);
      assert(weight->region != LogicalRegion::NO_REGION);
      parameters.push_back(weight);
    }

    op->map_output_tensors(*this);

    if (op->is_parallel_op()) {
      ((ParallelOp *)op)->create_input_partition(*this);
    }
  }

  // Check correctness
  for (size_t l = 0; l < operators.size(); l++) {
    Op *op = operators[l];
    for (int i = 0; i < op->numOutputs; i++) {
      assert(op->outputs[i]->owner_op == op);
      assert(op->outputs[i]->owner_idx == i);
      assert(op->outputs[i]->parallel_tensor_guid != 0);
    }
  }

  this->optimize_unnecessary_gradient_calculations();

  if (config.perform_fusion) {
    this->perform_fusion_optimizations();
  }

  Op *final_operator = get_final_operator();
  // FIXME: currently assume the final operator has exactly one output
  assert(final_operator->numOutputs == 1);
  this->print_operator_regions();

  this->create_label_tensor(loss_type);
  
  // init optimizer
  assert(optimizer != NULL);
  optimizer->init();

#ifdef FF_USE_NCCL
  if (config.computationMode == COMP_MODE_TRAINING) {
    this->initialize_nccl_communicators();
  }
#endif
}

void FFModel::zero_gradients(void) {
  for (int l = operators.size() - 1; l >= 0; l--) {
    operators[l]->zero_grad(*this);
  }
}

std::unordered_map<Op *, std::vector<std::pair<Op *, int>>>
    FFModel::get_bwd_edge_map() const {
  std::unordered_map<Op *, std::vector<std::pair<Op *, int>>> bwd_edge_map;
  for (auto const &op : this->operators) {
    for (int i = 0; i < op->numInputs; i++) {
      Op *src = (Op *)op->inputs[i]->owner_op;
      bwd_edge_map[src].push_back({op, op->inputs[i]->get_volume()});
    }
  }

  return bwd_edge_map;
};

void Op::prefetch(FFModel const &ff) {
  // TODO: perform prefetch for performance imporvement
}

// ========================================================
// class FFIterationConfig
// ========================================================
FFIterationConfig::FFIterationConfig() {
  seq_length = -1;
}

void FFIterationConfig::reset() {
  seq_length = -1;
}

};
