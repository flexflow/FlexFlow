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
#include "legion/legion_utilities.h"
#include "legion_parallel_tensor_shape.h"
#include "mapper.h"
#include "op-attrs/ops/noop.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "parallel_tensor_mapping.h"
#include "task_spec/task_argument_accessor.h"
#include "test_utils.h"
#include "utils/random_utils.h"
#include <dirent.h>
#include <queue>
#include <unordered_set>

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

/* FFModel::FFModel(FFConfig const &_config, */
/*                  ComputationGraph const &cg, */
/*                  TrainingPCG const &training_pcg, */
/*                  Optimizer const &_optimizer, */
/*                  LegionBacking const &_runtime_backing, */
/*                  EnableProfiling const &_enable_profiling, */
/*                  SimEnvFactory const &_sim_factory, */
/*                  TensorMapping const &_tensor_map) */
/*     : config(_config), computation_graph(cg), training_pcg(training_pcg), */
/*       optimizer(_optimizer), runtime_backing(_runtime_backing), */
/*       enable_profiling(_enable_profiling), sim_factory(_sim_factory), */
/*       tensor_map(_tensor_map) { */
/* ArgumentMap argmap; */
/* Rect<1> task_rect(Point<1>(0), */
/*                   Point<1>(config.workersPerNode * config.numNodes - 1));
 */
/* IndexSpaceT<1> task_is = runtime->create_index_space(ctx, task_rect); */

/* for (PointInRectIterator<1> it(task_rect); it(); it++) { */
/*   FFInitInfo info; */
/*   info.workSpaceSize = config.workSpaceSize; */
/*   info.allowTensorOpMathConversion =
 * config.allow_tensor_op_math_conversion; */
/*   argmap.set_point(*it, TaskArgument(&info, sizeof(FFInitInfo))); */
/* } */

// Init CUDA library on each worker
// IndexLauncher initLauncher(FF_INIT_TASK_ID,
//                            task_is,
//                            TaskArgument(NULL, 0),
//                            argmap,
//                            Predicate::TRUE_PRED,
//                            false /*must*/,
//                            0 /*mapper_id*/,
//                            FFConfig::DataParallelism_GPU);
// FutureMap fm = runtime->execute_index_space(ctx, initLauncher);
// fm.wait_all_results();
// int idx = 0;
// for (PointInRectIterator<1> it(task_rect); it(); it++) {
//   handlers[idx++] = fm.get_result<FFHandler>(*it);
// }
/* } */

/* using FullyExecutableArgSpec = variant<ConcreteArgSpec, CheckedTypedFuture>;
 */

/* struct ArgumentsConstructionState { */
/*   variant<Legion::TaskLauncher, Legion::IndexLauncher> launcher; */
/*   int num_futures; */
/*   Legion::Serializer sez; */
/*   TaskArgumentsFormat args_fmt; */
/* }; */

/* struct AddArgumentToTaskFunctor { */
/*   AddArgumentToTaskFunctor(ArgumentsConstructionState &state, slot_id slot) :
 * state(state), slot(slot) { } */

/*   ArgumentsConstructionState &state; */
/*   slot_id slot; */

/*   void operator()(ConcreteArgSpec const &a) { */
/*     size_t start = state.sez.get_used_bytes(); */
/*     a.serialize(state.sez); */
/*     size_t end = state.sez.get_used_bytes(); */
/*     state.args_fmt.insert({slot,
 * TaskArgumentFormat(a.get_type_tag().get_type_idx(), start, end)}); */
/*   } */

/*   void operator()(CheckedTypedFuture const &a) { */
/*     if (holds_alternative<Legion::TaskLauncher>(state.launcher)) { */
/*       get<Legion::TaskLauncher>(state.launcher).add_future(a.get_unsafe());
 */
/*     } else { */
/*       get<Legion::IndexLauncher>(state.launcher).add_future(a.get_unsafe());
 */
/*     } */
/*     state.args_fmt.insert({slot,
 * FutureArgumentFormat(a.get_type_tag().get_type_idx(), state.num_futures)});
 */
/*     state.num_futures++; */
/*   } */
/* }; */

/* static TaskArgumentFormat add_argument_to_task_arg(ArgumentsConstructionState
 * &state, */
/*                                                    slot_id slot, */
/*                                                    FullyExecutableArgSpec
 * const &arg_spec) { */
/*   visit(AddArgumentToTaskFunctor{state, slot}, arg_spec); */
/* } */

// TaskArgumentsFormat create_serializable_format(TensorArgsFormat const
// &tensor_args_format,
//                                                ConcreteArgsFormat const
//                                                &concrete_args_format,
//                                                FutureArgsFormat const
//                                                &future_args_format,
//                                                optional<IndexArgsFormat>
//                                                const &index_args_format =
//                                                nullopt);
/* TaskArgumentsFormat result; */
/* for (auto const &kv : concrete_args_format.fmts) { */
/*   result.insert(kv); */
/* } */
/* for (auto const &kv : future_args_format.fmts) { */
/*   result.insert(kv); */
/* } */
/* assert (!index_args_format.has_value()); */
/* for (parallel_tensor_guid_t const &guid :
 * keys(tensor_args_format.region_idxs)) { */
/*   region_idx_t region_idx = tensor_args_format.region_idxs.at_l(guid); */
/*   Legion::PrivilegeMode privs =
 * to_legion(tensor_args_format.privs_map.at(guid)); */
/*   DataType datatype = tensor_args_format.datatypes.at(guid); */
/*   result.insert(region_idx, privs, datatype); */
/* } */
/* for (auto const &kv : tensor_args_format.nonvariadic_slot_to_tensor) { */
/*   slot_id slot = kv.first; */
/*   parallel_tensor_guid_t guid = kv.second; */
/*   region_idx_t region_idx = tensor_args_format.region_idxs.at_l(guid); */
/*   result.insert(slot, region_idx); */
/* } */
/* for (auto const &kv : tensor_args_format.variadic_slot_to_tensor) { */
/*   slot_id slot = kv.first; */
/*   std::vector<parallel_tensor_guid_t> guids = kv.second; */
/*   std::vector<region_idx_t> region_idxs = transform(guids,
 * lookup_in_l(tensor_args_format.region_idxs)); */
/*   result.insert(slot, region_idxs); */
/* } */
/* return result; */
/* } */

// TaskReturnAccessor execute(TensorlessTaskInvocation const &invocation,
//                            /* TensorArgsFormat const &tensor_args_format, */
//                            ParallelComputationGraph const &pcg,
//                            RuntimeBacking const &backing,
//                            EnableProfiling enable_profiling) {
//   TaskSignature sig = get_signature(invocation.task_id);
//   TensorlessTaskBinding binding = invocation.binding;
//   /* TensorArgsFormat tensor_args_format = process_tensor_args(sig, pcg,
//   binding); */ ConcreteArgsFormat concrete_args_format =
//   process_concrete_args(binding); FutureArgsFormat future_args_format =
//   process_future_args(binding); TaskInvocationArgsFormat
//   task_invocation_args_format = process_task_invocation_args(binding,
//   enable_profiling, backing); assert
//   (get_args_of_type<CheckedTypedFutureMap>(binding).empty()); // currently we
//   don't handle these as I don't think they're used anywhere if
//   (binding.invocation_type == InvocationType::STANDARD) {
//     assert (get_args_of_type<IndexArgSpec>(binding).empty());
//     Legion::TaskArgument task_arg = as_task_argument(concrete_args_format,
//                                                      future_args_format,
//                                                      tensor_args_format);
//     TaskLauncher launcher(invocation.task_id, task_arg);
//     add_tensor_requirements(launcher, tensor_args_format);
//     Future returned_future = backing.execute_task(launcher);
//     return TaskReturnAccessor(sig.get_return_type(), returned_future);
//   } else if (binding.invocation_type == InvocationType::INDEX) {
//     parallel_tensor_guid_t index_space_determiner =
//     binding.domain_spec.value(); ParallelTensorBacking pt_backing =
//     backing.at(index_space_determiner); IndexArgsFormat index_args_format =
//     process_index_args(binding,
//                                                            backing.get_domain(pt_backing.parallel_is));
//     Legion::TaskArgument task_arg = as_task_argument(concrete_args_format,
//                                                      future_args_format,
//                                                      tensor_args_format,
//                                                      index_args_format);
//     IndexTaskLauncher launcher(invocation.task_id,
//                                pt_backing.parallel_is,
//                                task_arg,
//                                as_argument_map(index_args_format),
//                                Predicate::TRUE_PRED,
//                                false /*must*/,
//                                0 /*mapper_id*/,
//                                pt_backing.mapping_id.value()
//                                );
//     add_tensor_requirements(launcher, tensor_args_format);
//     FutureMap returned_future = backing.execute_task(launcher);
//     return TaskReturnAccessor(sig.get_return_type(), returned_future);
//   }
// }

/* void init_operators(FFModel const &ff) { */
/*   ff.execute(init_operators(ff.training_pcg)); */
/* } */

/* void forward(FFModel const &ff, int seq_length) { */
/*   iter_config.seq_length = seq_length; */
/*   ff.execute(forwad(ff.training_pcg)); */
/* } */

/* TypedFuture<PerfMetrics> compute_metrics(FFModel const &ff, */
/*                                          PerfMetrics const &metrics) { */
/*   TaskReturnAccessor acc = */
/*       ff.execute(compute_metrics(ff.training_pcg, metrics)); */
/*   return acc.get_returned_future<PerfMetrics>().get(); */
/* } */

/* void backward(FFModel const &ff, int seq_length, PerfMetrics const &metrics)
 * { */
/*   iter_config.seq_length = seq_length; */
/*   assert(ff.config.computationMode == ComputationMode::TRAINING); */
/*   // Compute metrics */
/*   compute_metrics(ff, metrics).get(); // TODO FIXME @lockshaw actually do */
/*                                       // something with the computed metrics
 */
/*   // Compute the gradients of the final operator wrt loss */
/*   ff.execute(backward(ff.training_pcg)); */
/*   /1* Op const *final_operator = get_final_operator(); *1/ */
/*   /1* assert(final_operator->numOutputs == 1); *1/ */
/*   /1* loss_op->backward(this, final_operator->outputs[0], */
/*    * parallel_label_tensor.value()); *1/ */
/*   /1* // Perform backpropagation *1/ */
/*   /1* // std::set<LogicalRegion> resetedInputGrads; *1/ */
/*   /1* for (int l = operators.size() - 1; l >= 0; l--) { *1/ */
/*   /1*   // TODO: If operator serves for metrics and for further prop *1/ */
/*   /1*   // if(l == metrics_input && metrics_input < (int)operators.size()-1)
 * *1/ */
/*   /1*   //  continue; *1/ */
/*   /1*   operators[l]->backward(*this); *1/ */
/*   /1* } *1/ */
/* } */

/* void update(FFModel const &ff) { */
/*   ff.execute(update(ff.pcg, ff.optimizer)); */
/* } */

/* operator_guid_t get_final_operator(FFModel const &ff) { */
/*   operator_guid_t final_op_id = get_only(get_sinks(ff.pcg.graph)); */
/*   // assert that the final operator has exactly one output */
/*   Operator op = ff.pcg.at(final_op_id); */
/*   assert(get_num_outputs(op) == 1); */
/*   return final_op_id; */
/* } */

/* void FFModel::compile(Optimizer *_optimizer, */
/*                       LossType loss_type, */
/*                       std::vector<MetricsType> const &metrics, */
/*                       CompMode comp_mode) { */
/*   optimizer = _optimizer; */
/*   compile(loss_type, metrics, comp_mode); */
/* } */

/* MachineView */
/*     get_basic_data_parallel_machine_view(MachineSpecification const &spec) {
 */
/*   gpu_id_t start = gpu_id_t(0); */
/*   gpu_id_t stop = gpu_id_t(spec.num_nodes * spec.workersPerNode); */
/*   return make_1d_machine_view(start, stop, 1); */
/* } */

/* MachineView get_basic_data_parallel_machine_view(FFConfig const &config) { */
/*   gpu_id_t start = gpu_id_t(0); */
/*   gpu_id_t stop = gpu_id_t(config.numNodes * config.workersPerNode); */
/*   return make_1d_machine_view(start, stop, 1); */
/* } */

/* static ParallelTensorShape get_parallel_tensor_shape(Tensor const &tensor) {
 */
/*   int num_dims = tensor->num_dims(); */
/*   std::vector<ParallelDim> dims; */
/*   for (int j = 0; j < num_dims; j++) { */
/*     dims.emplace_back(tensor->dims[j], 1, -1, false); */
/*   } */
/*   dims.emplace_back(1, 1, -1, true); */
/*   ParallelTensorShape shape = {dims, tensor->data_type}; */
/*   return shape; */
/* } */

/* void FFModel::print_operator_regions() const { */
/*   for (size_t i = 0; i < operators.size(); i++) { */
/*     Op *op = operators[i]; */
/*     printf("operator[%zu]: type(%d)\n", i, operators[i]->op_type); */
/*     for (int j = 0; j < op->numInputs; j++) { */
/*       LogicalRegion handle = op->inputs[j]->region; */
/*       printf("inputs[%d] region(%d,%d,%d)\n", */
/*              j, */
/*              handle.get_index_space().get_id(), */
/*              handle.get_field_space().get_id(), */
/*              handle.get_tree_id()); */
/*     } */
/*     for (int j = 0; j < op->numOutputs; j++) { */
/*       LogicalRegion handle = op->outputs[j]->region; */
/*       printf("outputs[%d] region(%d,%d,%d)\n", */
/*              j, */
/*              handle.get_index_space().get_id(), */
/*              handle.get_field_space().get_id(), */
/*              handle.get_tree_id()); */
/*     } */
/*   } */
/* } */

/* void FFModel::create_label_tensor(LossType loss_type) { */
/*   Op const *final_operator = get_final_operator(); */

/*   std::vector<ParallelDim> p_dims = */
/*       final_operator->outputs[0]->get_shape().dims; */

/*   std::vector<size_t> dims; */
/*   // FIXME: Currently assume 1st input for 1st operator = batch_size */
/*   for (ParallelDim const &dim : p_dims) { */
/*     if (!dim.is_replica_dim) { */
/*       dims.push_back(dim.size); */
/*     } */
/*   } */

/*   DataType label_type = DT_FLOAT; */
/*   if (loss_type == LOSS_SPARSE_CATEGORICAL_CROSSENTROPY) { */
/*     // assign dims[num_dims-1] = 1 for sparse categorical labels */
/*     assert(p_dims[0].degree == 1); */
/*     p_dims[0].size = 1; */
/*     dims[0] = 1; */
/*     label_type = DT_INT32; */
/*   } */

/*   LegionParallelTensorShape label_p_shape = {p_dims, label_type}; */
/*   LegionTensorShape label_shape = {dims, label_type}; */

/*   // create label tensor */
/*   label_tensor = */
// create_tensor(label_shape, NULL, 0 /*idx*/, false /*create_grad*/);
/*   parallel_label_tensor = create_parallel_tensor(label_p_shape); */
/*   label_tensor.value()->parallel_tensor = parallel_label_tensor; */
/*   parallel_label_tensor.value()->machine_view = */
/*       final_operator->outputs[0]->machine_view; */
/*   map_tensor(parallel_label_tensor.value(), */
/*              parallel_label_tensor.value()->owner_op, */
/*              this->config.legion_config, */
/*              this->index_space_mgr); */
/* } */

/* void FFModel::execute_graph_optimize() { */
/*   FFModel *model = this; */
/*   Context ctx = config.legion_config.lg_ctx; */
/*   Runtime *runtime = config.legion_config.lg_hlr; */
/*   TaskLauncher launcher(GRAPH_OPTIMIZE_TASK_ID, */
/*                         TaskArgument(&model, sizeof(FFModel *))); */
/*   Future future = runtime->execute_task(ctx, launcher); */

/*   PCG::GraphOptimalViewSerialized ret = */
/*       future.get_result<PCG::GraphOptimalViewSerialized>(); */
/*   Deserializer dez(ret.data, ret.total_bytes); */
/*   // Reconstruct operators */
/*   PCG::Graph *best_graph = new PCG::Graph(this); */
/*   std::unordered_map<PCG::Node, MachineView> optimal_views; */
/*   deserialize_graph_optimal_view(dez, best_graph, optimal_views); */
/*   operators.clear(); */
/*   convert_graph_to_operators(best_graph, optimal_views); */
/*   best_graph->print_dot(); */
/*   delete best_graph; */

/*   this->populate_tensor_to_parallel_tensor_mapping(); */
/* } */

/* void FFModel::compile(LossType loss_type, */
/*                       std::vector<MetricsType> const &metrics, */
/*                       CompMode comp_mode) { */
/*   if (metrics_input == -1) { */
/*     metrics_input = operators.size() - 1; */
/*   } */
/*   Context ctx = config.legion_config.lg_ctx; */
/*   Runtime *runtime = config.legion_config.lg_hlr; */
/*   config.computationMode = comp_mode; */
/*   // if (config.import_strategy_file.length() > 0) { */
/*   //   load_strategies_from_file(config.import_strategy_file, */
/*   //   config.strategies); */
/*   // } */
/*   //  Construct operators from layers */
/*   if (config.only_data_parallel) { */
/*     fprintf(stderr, */
/*             "Note: only_data_parallel is specified, FlexFlow compiles a " */
/*             "data-parallel PCG.\n"); */
/*   } */
/*   this->create_operators_from_layers(); */

/*   // Launch the graph optimize task */
/*   this->execute_graph_optimize(); */

/*   bool repl_labels = (operators[operators.size() - 1]->op_type ==
 * OP_AGG_SPEC); */
/*   loss_op = {loss_type, repl_labels}; */
/*   metrics_op = {loss_type, metrics}; */

/*   // Init performance metrics */
/*   TaskLauncher launcher(UPDATE_METRICS_TASK_ID, */
/*                         TaskArgument(&metrics_op.value(), sizeof(Metrics)));
 */
/*   current_metrics = runtime->execute_task(ctx, launcher); */

/*   if (config.enable_inplace_optimizations) { */
/*     this->perform_inplace_optimizations(); */
/*   } */

/*   for (Op *op : this->operators) { */
/*     for (ParallelTensor const &input : op->inputs) { */
/*       assert(input->owner_op != NULL); */
/*     } */

/*     for (ParallelTensor const &weight : op->weights) { */
/*       assert(weight->owner_op != NULL); */
/*       assert(weight->region != LogicalRegion::NO_REGION); */
/*       parameters.push_back(weight); */
/*     } */

/*     op->map_output_tensors(*this); */

/*     if (op->is_parallel_op()) { */
/*       ((ParallelOp *)op)->create_input_partition(*this); */
/*     } */
/*   } */

/*   // Check correctness */
/*   for (size_t l = 0; l < operators.size(); l++) { */
/*     Op *op = operators[l]; */
/*     for (int i = 0; i < op->numOutputs; i++) { */
/*       assert(op->outputs[i]->owner_op == op); */
/*       assert(op->outputs[i]->owner_idx == i); */
/*       assert(op->outputs[i]->parallel_tensor_guid != 0); */
/*     } */
/*   } */

/*   this->optimize_unnecessary_gradient_calculations(); */

/*   if (config.perform_fusion) { */
/*     this->perform_fusion_optimizations(); */
/*   } */

/*   Op *final_operator = get_final_operator(); */
/*   // FIXME: currently assume the final operator has exactly one output */
/*   assert(final_operator->numOutputs == 1); */
/*   this->print_operator_regions(); */

/*   this->create_label_tensor(loss_type); */

/*   // init optimizer */
/*   assert(optimizer != NULL); */
/*   optimizer->init(); */

/* #ifdef FF_USE_NCCL */
/*   if (config.computationMode == COMP_MODE_TRAINING) { */
/*     this->initialize_nccl_communicators(); */
/*   } */
/* #endif */
/* } */

/* void FFModel::zero_gradients(void) { */
/*   for (int l = operators.size() - 1; l >= 0; l--) { */
/*     operators[l]->zero_grad(*this); */
/*   } */
/* } */

/* // ======================================================== */
/* // class FFIterationConfig */
/* // ======================================================== */
/* FFIterationConfig::FFIterationConfig() { */
/*   seq_length = -1; */
/* } */

/* void FFIterationConfig::reset() { */
/*   seq_length = -1; */
/* } */

}; // namespace FlexFlow
