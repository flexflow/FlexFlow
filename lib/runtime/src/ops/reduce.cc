#include "reduce.h"
#include "kernels/reduce_kernels.h"
#include "legion/legion_utilities.h"
#include "utils/exception.decl.h"
#include "utils/hash-utils.h"
#include "op-attrs/get_output_shape.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

using namespace FlexFlow::Kernels::Reduce;

bool operator==(ReduceParams const &lhs, ReduceParams const &rhs) {
  return (lhs.axes == rhs.axes) && (lhs.keepdims == rhs.keepdims);
}

bool ReduceParams::is_valid(ParallelTensorShape const &input) const {
  for (size_t i = 0; i < axes.size(); i++) {
    if (axes[i] >= input.num_dims) {
      return false;
    }
  }
  return input.is_valid();
}

ReduceParams Reduce::get_params() const {
  ReduceParams params;
  params.axes.clear();
  for (int i = 0; i < num_axes; i++) {
    params.axes.push_back(this->axes[i]);
  }
  params.keepdims = keepdims;
  return params;
}

enum Slots {INPUT, OUTPUT, ATTRS, PROFILING, REDUCE, PER_DEVICE_STATE, HANDLE};

OpTaskInvocation init(TransposeAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(HANDLE, ff_handle());.
  binding.bind_arg(ATTRS, attrs);

  return {REDUCE_INIT_TASK_ID, binding};
}

static DeviceSpecific<ReducePerDeviceState> init_task_impl(TaskArgumentAccessor const &acc) {
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  auto attrs = acc.get_argument<ReduceAttrs>(ATTRS);
  OperatorType = attrs.op_type;
  //Note: How to set the reduction size?
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffReduceTensorDescriptor_t reduceDesc;
  size_t reduction_size
  DeviceSpecific<ReducePerDeviceState> per_device_state = acc.create_device_specific<ReducePerDeviceState>(init_kernel(handle, input, output, reduce_desc, op_type, reduction_size));
  return per_device_state;
}

static DeviceSpecific<TransposePerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

plate <>
void register_task<TRANSPOSE_INIT_TASK_ID>() {
    OpTaskSignature init(OpTaskType::INIT)

    init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);
    init.add_arg_slot<ReduceAttrs>(ATTRS);

    register_task(REDUCE_INIT_TASK_ID, "Reduce::init", init, init_task);
} 

//Note: forward_kernel only needs ReducePerDeviceState, input, output
OpTaskInvocation forward(ReduceAttrs const & attrs) {
  OpTaskBinding binding;

  bind.bind_arg(PER_DEVICE_STATE, per_device_op_state<ReducePerDeviceState>());
  bind.bind_arg(PROFILING, profiling_tensor());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {REDUCE_FWD_TASK_ID, binding};
}


static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  NOT_IMPLEMENTED();
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

template <>
void register_task<REDUCE_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FORWARD);

  fwd.add_unchecked_arg_slot<PerDeviceOpState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<ProfilingTensor>(PROFILING);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(REDUCE_FWD_TASK_ID, "Reduce::forward", fwd, forward_task);
}

OpTaskInvocation backward(ReduceAttrs const & attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {REDUCE_BWD_TASK_ID, binding};
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  NOT_IMPLEMENTED();
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

template <>
void register_task<REDUCE_BWD_TASK_ID>() {
    OpTaskSignature bwd = infer_bwd_signature(get_op_signature(REDUCE_FWD_TASK_ID));

    reister_task(REDUCE_BWD_TASK_ID, "Reduce::backward", bwd, backward_task);
}

// Tensor FFModel::reduce_sum(OperatorType op,
//                            const Tensor input,
//                            std::vector<int> const &_axes,
//                            bool keepdims,
//                            char const *name) {
//   Layer *rd = new Layer(this,
//                         op,
//                         DT_FLOAT,
//                         name,
//                         1 /*input*/,
//                         0 /*weights*/,
//                         1 /*outputs*/,
//                         input);
//   // Use Legion indexing to store axes
//   std::vector<int> axes;
//   for (size_t i = 0; i < _axes.size(); i++) {
//     axes.push_back(input->num_dims - 1 - _axes[i]);
//   }
//   int dims[MAX_TENSOR_DIM];
//   int numdim = input->num_dims;
//   if (keepdims) {
//     for (int i = 0; i < input->num_dims; i++) {
//       dims[i] = input->dims[i];
//     }
//     for (size_t i = 0; i < axes.size(); i++) {
//       dims[axes[i]] = 1;
//     }
//   } else {
//     numdim = 0;
//     for (int i = 0; i < input->num_dims; i++) {
//       bool reduced = false;
//       for (size_t j = 0; j < axes.size(); j++) {
//         if (axes[j] == i) {
//           reduced = true;
//         }
//       }
//       if (!reduced) {
//         dims[numdim++] = input->dims[i];
//       }
//     }
//     assert(numdim + axes.size() == input->num_dims);
//   }
//   rd->outputs[0] = create_tensor_legion_ordering(
//       numdim, dims, input->data_type, rd, 0, true /*create_grad*/);
//   rd->add_int_vector_property("legion_axes", axes);
//   rd->add_int_property("keepdims", keepdims);
//   layers.push_back(rd);
//   return rd->outputs[0];
// }

// Tensor FFModel::reduce_sum(const Tensor input,
//                            std::vector<int> const &_axes,
//                            bool keepdims,
//                            char const *name) {
//   return this->reduce(OP_REDUCE_SUM, input, _axes, keepdims, name);
// }

// Tensor FFModel::reduce_mean(const Tensor input,
//                             std::vector<int> const &_axes,
//                             bool keepdims,
//                             char const *name) {
//   return this->reduce(OP_REDUCE_MEAN, input, _axes, keepdims, name);
// }

// Op *Reduce::create_operator_from_layer(
//     FFModel &model,
//     Layer const *layer,
//     std::vector<ParallelTensor> const &inputs) {
//   std::vector<int> axes;
//   long long value;
//   layer->get_int_vector_property("legion_axes", axes);
//   layer->get_int_property("keepdims", value);
//   bool keepdims = value;
//   return new Reduce(
//       model, layer->op_type, inputs[0], axes, keepdims, layer->name);
// }

// Reduce::Reduce(FFModel &model,
//                ReduceParams const &params,
//                const ParallelTensor input,
//                char const *name)
//     : Reduce(model, params.op_type, input, params.axes, params.keepdims, name) {
// }

// Reduce::Reduce(FFModel &model,
//                OperatorType _op_type,
//                const ParallelTensor input,
//                std::vector<int> const &_axes,
//                bool _keepdims,
//                char const *name)
//     : Op(model,
//          _op_type,
//          input->data_type,
//          name,
//          1 /*inputs*/,
//          0 /*weights*/,
//          1 /*outputs*/,
//          input),
//       num_axes(_axes.size()), keepdims(_keepdims) {
//   for (size_t i = 0; i < num_axes; i++) {
//     axes[i] = _axes[i];
//   }
//   int num_dims = input->num_dims;
//   ParallelDim dims[MAX_TENSOR_DIM];
//   if (keepdims) {
//     num_dims = input->num_dims;
//     for (int i = 0; i < num_dims; i++) {
//       dims[i] = input->dims[i];
//     }
//     for (int i = 0; i < num_axes; i++) {
//       // Currently assume that we cannot parallelize along reduced dims
//       assert(dims[axes[i]].degree == 1);
//       dims[axes[i]].size = 1;
//     }
//   } else {
//     num_dims = 0;
//     for (int i = 0; i < input->num_dims; i++) {
//       bool reduced = false;
//       for (int j = 0; j < num_axes; j++) {
//         if (axes[j] == i) {
//           reduced = true;
//         }
//       }
//       if (!reduced) {
//         dims[num_dims++] = input->dims[i];
//       } else {
//         // Currently assume that we cannot parallelize along reduced dims
//         assert(input->dims[i].degree == 1);
//         assert(input->dims[i].parallel_idx == -1);
//       }
//     }
//   }
//   outputs[0] = model.create_parallel_tensor_legion_ordering(
//       num_dims, dims, input->data_type, this);
// }

// void Reduce::init(FFModel const &ff) {
//   assert(check_output_input_weight_same_parallel_is());
//   parallel_is = outputs[0]->parallel_is;
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   set_argumentmap_for_init(ff, argmap);
//   IndexLauncher launcher(REDUCE_INIT_TASK_ID,
//                          parallel_is,
//                          TaskArgument(this, sizeof(Reduce)),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          outputs[0]->machine_view.hash());
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region));
//   launcher.add_field(0, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     WRITE_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region));
//   launcher.add_field(1, FID_DATA);
//   FutureMap fm = runtime->execute_index_space(ctx, launcher);
//   fm.wait_all_results();
//   set_opmeta_from_futuremap(ff, fm);
// };

// PerDeviceOpState *Reduce::init_task(Task const *task,
//                                     std::vector<PhysicalRegion> const &regions,
//                                     Context ctx,
//                                     Runtime *runtime) {
//   Reduce *rd = (Reduce *)task->args;
//   FFHandler handle = *((FFHandler *)task->local_args);
//   GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
//       DT_FLOAT, regions[0], task->regions[0], FID_DATA, ctx, runtime);
//   GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
//       DT_FLOAT, regions[1], task->regions[1], FID_DATA, ctx, runtime);
//   ReduceMeta *m = new ReduceMeta(handle, rd, input.domain);
//   return m;
// }

// void Reduce::forward(FFModel const &ff) {
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   set_argumentmap_for_forward(ff, argmap);
//   IndexLauncher launcher(REDUCE_FWD_TASK_ID,
//                          parallel_is,
//                          TaskArgument(nullptr, false),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          outputs[0]->machine_view.hash());
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region));
//   launcher.add_field(0, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     WRITE_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region));
//   launcher.add_field(1, FID_DATA);
//   runtime->execute_index_space(ctx, launcher);
// }

// void Reduce::forward_task(Task const *task,
//                           std::vector<PhysicalRegion> const &regions,
//                           Context ctx,
//                           Runtime *runtime) {
//   assert(regions.size() == 2);
//   assert(task->regions.size() == 2);
//   ReduceMeta const *m = *((ReduceMeta **)task->local_args);
//   GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
//       DT_FLOAT, regions[0], task->regions[0], FID_DATA, ctx, runtime);
//   GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
//       DT_FLOAT, regions[1], task->regions[1], FID_DATA, ctx, runtime);

//   forward_kernel_wrapper(m, input, output);
// }

// void Reduce::backward(FFModel const &ff) {
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   set_argumentmap_for_backward(ff, argmap);
//   IndexLauncher launcher(REDUCE_BWD_TASK_ID,
//                          parallel_is,
//                          TaskArgument(nullptr, 0),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          outputs[0]->machine_view.hash());
//   // regions[0](I): output_grad
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region_grad));
//   launcher.add_field(0, FID_DATA);
//   // regions[1](I/O): input_grad
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_WRITE,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region_grad));
//   launcher.add_field(1, FID_DATA);
//   runtime->execute_index_space(ctx, launcher);
// }

// void Reduce::backward_task(Task const *task,
//                            std::vector<PhysicalRegion> const &regions,
//                            Context ctx,
//                            Runtime *runtime) {
//   assert(regions.size() == 2);
//   assert(task->regions.size() == 2);
//   ReduceMeta const *m = *((ReduceMeta **)task->local_args);
//   GenericTensorAccessorR output_grad = helperGetGenericTensorAccessorRO(
//       DT_FLOAT, regions[0], task->regions[0], FID_DATA, ctx, runtime);
//   GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
//       DT_FLOAT, regions[1], task->regions[1], FID_DATA, ctx, runtime);
//   backward_kernel_wrapper(m, output_grad, input_grad);
// }

// bool Reduce::measure_operator_cost(Simulator *sim,
//                                    MachineView const &mv,
//                                    CostMetrics &cost_metrics) const {
//   ParallelTensorBase sub_input, sub_output;
//   if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
//     return false;
//   }
//   if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
//     return false;
//   }
//   ReduceMeta *m = new ReduceMeta(sim->handler, this, sub_input.get_domain());
//   sim->free_all();
//   float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
//   assert(input_ptr != NULL);
//   cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
//   GenericTensorAccessorR input_acc(
//       inputs[0]->data_type, sub_input.get_domain(), input_ptr);

//   float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
//   assert(output_ptr != NULL);
//   cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
//   GenericTensorAccessorW output_acc(
//       outputs[0]->data_type, sub_output.get_domain(), output_ptr);

//   assert(m->profiling == false);

//   std::function<void()> forward, backward;
//   forward = [&] { forward_kernel_wrapper(m, input_acc, output_acc); };
//   if (sim->computationMode == COMP_MODE_TRAINING) {
//     float *input_grad_ptr =
//         (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
//     assert(input_grad_ptr != NULL);
//     cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
//     GenericTensorAccessorW input_grad_acc(
//         inputs[0]->data_type, sub_input.get_domain(), input_grad_ptr);

//     float *output_grad_ptr =
//         (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
//     assert(output_grad_ptr != NULL);
//     cost_metrics.outputs_memory +=
//         cost_metrics.total_mem_diff_from(sim->offset);
//     GenericTensorAccessorR output_grad_acc(
//         outputs[0]->data_type, sub_output.get_domain(), output_grad_ptr);

//     backward = [&] {
//       backward_kernel_wrapper(m, output_grad_acc, input_grad_acc);
//     };
//   }

//   inner_measure_operator_cost(sim, forward, backward, cost_metrics);

//   if (sim->computationMode == COMP_MODE_TRAINING) {
//     printf("[Measure Reduce] name(%s) forward_time(%.4lf) "
//            "backward_time(%.4lf)\n",
//            name,
//            cost_metrics.forward_time,
//            cost_metrics.backward_time);
//   } else {
//     printf("[Measure Reduce] name(%s) forward_time(%.4lf)\n",
//            name,
//            cost_metrics.forward_time);
//   }

//   return true;
// }

// void Reduce::serialize(Legion::Serializer &sez) const {
//   ReduceParams params = get_params();
//   sez.serialize(params.op_type);
//   sez.serialize(params.axes.size());
//   for (size_t i = 0; i < params.axes.size(); i++) {
//     sez.serialize(params.axes[i]);
//   }
//   sez.serialize(params.keepdims);
// }

// using PCG::Node;
// Node Reduce::deserialize(FFModel &ff,
//                          Legion::Deserializer &dez,
//                          ParallelTensor inputs[],
//                          int num_inputs) {
//   assert(num_inputs == 1);
//   OperatorType op_type;
//   size_t axes_size;
//   bool keepdims;
//   std::vector<int> axes;
//   dez.deserialize(op_type);
//   dez.deserialize(axes_size);
//   for (size_t i = 0; i < axes_size; i++) {
//     int dim_idx;
//     dez.deserialize(dim_idx);
//     axes.push_back(dim_idx);
//   }
//   dez.deserialize(keepdims);
//   return ff.get_or_create_node<Reduce>(inputs[0], {axes, op_type, keepdims});
// }

// Op *Reduce::materialize(FFModel &ff,
//                         ParallelTensor inputs[],
//                         int num_inputs) const {
//   ReduceParams params = get_params();
//   return new Reduce(ff, params, inputs[0], this->name);
// }

// }; // namespace FlexFlow

// namespace std {
// size_t hash<FlexFlow::ReduceParams>::operator()(
//     FlexFlow::ReduceParams const &params) const {
//   size_t key = 0;
//   hash_combine(key, params.op_type);
//   hash_combine(key, params.axes.size());
//   for (int n : params.axes) {
//     hash_combine(key, n);
//   }
//   hash_combine(key, params.keepdims);
//   return key;
// }
}; // namespace std
