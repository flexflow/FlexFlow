#include "flexflow/ops/stop_grad.h"
#include "flexflow/model.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

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

Tensor FFModel::stopgrad(const Tensor x, char const *name) {
  Layer *ele = new Layer(
      this, OP_STOPGRAD, name, 1 /*inputs*/, 0 /*weights*/, 1 /*outputs*/, x);
  DataType dtype = x->data_type;
  int numdims = x->num_dims;
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdims; i++) {
    dims[i] = x->dims[i];
  }
  ele->outputs[0] = create_tensor_legion_ordering(
      numdims, dims, dtype, ele, 0, false /*create_grad*/);
  layers.push_back(ele);
  return ele->outputs[0];
}

Op *StopGrad::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  return new StopGrad(model, inputs[0], layer->name);
}

StopGradParams StopGrad::get_params() const {
  StopGradParams params;
  return params;
}

bool StopGradParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

bool operator==(StopGradParams const &lhs, StopGradParams const &rhs) {
  return true;
}

StopGrad::StopGrad(FFModel &model, const ParallelTensor x, char const *name)
    : Op(model,
         OP_STOPGRAD,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         x) {
  numOutputs = 1;
  int numdim = x->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = x->dims[i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, inputs[0]->data_type, this);
}

StopGrad::StopGrad(FFModel &model,
                   StopGradParams const &params,
                   const ParallelTensor input,
                   char const *name)
    : StopGrad(model, input, name) {}

void StopGrad::map_output_tensors(FFModel &ff) {
  Op::map_output_tensors(ff);
}

void StopGrad::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher init_launcher(STOPGRAD_INIT_TASK_ID,
                              parallel_is,
                              TaskArgument(this, sizeof(StopGrad)),
                              argmap,
                              Predicate::TRUE_PRED,
                              false /*must*/,
                              0 /*mapper_id*/,
                              outputs[0]->machine_view.hash());
  init_launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                         0 /*projection id*/,
                                                         READ_WRITE,
                                                         EXCLUSIVE,
                                                         inputs[0]->region));
  init_launcher.add_field(0, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *StopGrad::init_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  StopGrad *eu = (StopGrad *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  StopGradMeta *m = new StopGradMeta(handle);
  m->data_type = eu->outputs[0]->data_type;
  // Input and output should have the same data type
  assert(eu->outputs[0]->data_type == eu->inputs[0]->data_type);
  m->profiling = eu->profiling;
  std::strcpy(m->op_name, eu->name);
  return m;
}

void StopGrad::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(STOPGRAD_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  assert(outputs[0]->part == inputs[0]->part);
  assert(outputs[0]->region == inputs[0]->region);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void StopGrad::forward_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  StopGradMeta const *m = *((StopGradMeta **)task->local_args);
  if (m->data_type == DT_FLOAT) {
    forward_task_with_type<float>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_DOUBLE) {
    forward_task_with_type<double>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT32) {
    forward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT64) {
    forward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in StopGrad forward");
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/
template <typename DT>
void StopGrad::forward_task_with_type(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  // const StopGrad* ele = (const StopGrad*) task->args;
  StopGradMeta const *m = *((StopGradMeta **)task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  const DT *input_ptr = NULL;
  DT *output_ptr = NULL;
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  output_ptr = helperGetTensorPointerRW<DT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  input_ptr = output_ptr;

  StopGrad::forward_kernel_wrapper<DT>(
      m, input_ptr, output_ptr, input_domain.get_volume());
}

void StopGrad::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(STOPGRAD_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  assert(inputs[0]->part == outputs[0]->part);
  assert(inputs[0]->part_grad == outputs[0]->part_grad);
  // regions[2](I): output_grad
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[3](I): output_grad
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void StopGrad::backward_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  StopGradMeta const *m = *((StopGradMeta **)task->local_args);
  if (m->data_type == DT_FLOAT) {
    backward_task_with_type<float>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_DOUBLE) {
    backward_task_with_type<double>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT32) {
    backward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT64) {
    backward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Embedding forward");
  }
}

/*
  regions[0](I): input
  regions[1](I/O): input_grad
  regions[2](I): output
  regions[3](I): output_grad
*/
template <typename DT>
void StopGrad::backward_task_with_type(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  // const StopGrad* ele = (const StopGrad*) task->args;
  StopGradMeta const *m = *((StopGradMeta **)task->local_args);
  const DT *input_ptr = NULL, *output_ptr = NULL, *output_grad_ptr = NULL;
  DT *input_grad_ptr = NULL;
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Domain input_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(input_grad_domain == input_domain);
  input_ptr = helperGetTensorPointerRO<DT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  input_grad_ptr = helperGetTensorPointerRW<DT>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  output_ptr = input_ptr;
  output_grad_ptr = input_grad_ptr;
  StopGrad::backward_kernel_wrapper<DT>(m,
                                        input_ptr,
                                        input_grad_ptr,
                                        output_ptr,
                                        output_grad_ptr,
                                        input_domain.get_volume());
}

void StopGrad::serialize(Legion::Serializer &sez) const {}

bool StopGrad::measure_operator_cost(Simulator *sim,
                                     MachineView const &mv,
                                     CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_input;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  StopGradMeta *m = sim->stop_grad_meta;
  sim->free_all();
  float *input_ptr =
      (float *)sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
  assert(input_ptr != NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = NULL;
  output_ptr = input_ptr;
  assert(output_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  assert(m->profiling == false);

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(m, input_ptr, output_ptr, sub_output.get_volume());
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr =
        (float *)sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
    assert(input_grad_ptr != NULL);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *output_grad_ptr = NULL;
    output_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), outputs[0]->data_type);
    assert(output_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    backward = [&] {
      backward_kernel_wrapper(m,
                              input_ptr,
                              input_grad_ptr,
                              output_ptr,
                              output_grad_ptr,
                              sub_output.get_volume());
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure Elewise Unary] name(%s) num_elements(%zu) "
                      "forward_time(%.4lf) backward_time(%.4lf)\n",
                      name,
                      sub_output.get_volume(),
                      cost_metrics.forward_time,
                      cost_metrics.backward_time);
  } else {
    log_measure.debug("[Measure Elewise Unary] name(%s) num_elements(%zu) "
                      "forward_time(%.4lf)\n",
                      name,
                      sub_output.get_volume(),
                      cost_metrics.forward_time);
  }
  return true;
}

using PCG::Node;
/*static*/
Node StopGrad::deserialize(FFModel &ff,
                           Legion::Deserializer &dez,
                           ParallelTensor inputs[],
                           int num_inputs) {
  assert(num_inputs == 1);

  StopGradParams params;
  return ff.get_or_create_node<StopGrad>(inputs[0], params);
}

Op *StopGrad::materialize(FFModel &ff,
                          ParallelTensor inputs[],
                          int num_inputs) const {
  assert(num_inputs == 1);
  return new StopGrad(ff, inputs[0], this->name);
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::StopGradParams>::operator()(
    FlexFlow::StopGradParams const &params) const {
  size_t key = 0;
  return key;
}
}; // namespace std
