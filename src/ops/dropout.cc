#include "flexflow/ops/dropout.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/dropout_kernels.h"
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
using Legion::InlineLauncher;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;
using PCG::Node;

using namespace FlexFlow::Kernels::Dropout;

Tensor FFModel::dropout(const Tensor input,
                        float rate,
                        unsigned long long seed,
                        char const *name) {
  // seed = 0 is preserved as None, so we use a random seed
  if (seed == 0) {
    seed = std::rand();
  }
  Layer *dropout = new Layer(this,
                             OP_DROPOUT,
                             DT_FLOAT,
                             name,
                             1 /*inputs*/,
                             0 /*weights*/,
                             1 /*outputs*/,
                             input);
  int numdims = input->num_dims;
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdims; i++) {
    dims[i] = input->dims[i];
  }
  dropout->outputs[0] = create_tensor_legion_ordering(
      numdims, dims, DT_FLOAT, dropout, 0, true /*create_grad*/);
  dropout->add_float_property("rate", rate);
  dropout->add_int_property("seed", seed);
  layers.push_back(dropout);
  return dropout->outputs[0];
}

Op *Dropout::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("seed", value);
  int seed = value;
  float rate;
  layer->get_float_property("rate", rate);
  return new Dropout(model, inputs[0], rate, seed, layer->name);
}

DropoutParams Dropout::get_params() const {
  DropoutParams params;
  params.rate = this->rate;
  params.seed = this->seed;
  return params;
}

bool DropoutParams::is_valid(ParallelTensorShape const &) const {
  // dropout is always valid
  return true;
}

bool operator==(DropoutParams const &lhs, DropoutParams const &rhs) {
  return lhs.rate == rhs.rate && lhs.seed == rhs.seed;
}

Dropout::Dropout(FFModel &model,
                 const ParallelTensor _input,
                 float _rate,
                 unsigned long long _seed,
                 char const *name)
    : Op(model,
         OP_DROPOUT,
         DT_FLOAT,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         _input),
      rate(_rate), seed(_seed) {
  // Set output shape
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < _input->num_dims; i++) {
    dims[i] = _input->dims[i];
  }
  numOutputs = 1;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, dims, DT_FLOAT, this);
}

Dropout::Dropout(FFModel &model,
                 Dropout const &other,
                 const ParallelTensor input)
    : Dropout(model, input, other.rate, other.seed, other.name) {}

Dropout::Dropout(FFModel &model,
                 DropoutParams const &params,
                 const ParallelTensor input,
                 char const *name)
    : Dropout(model, input, params.rate, params.seed, name) {}

void Dropout::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher init_launcher(DROPOUT_INIT_TASK_ID,
                              parallel_is,
                              TaskArgument(this, sizeof(Dropout)),
                              argmap,
                              Predicate::TRUE_PRED,
                              false /*must*/,
                              0 /*mapper_id*/,
                              outputs[0]->machine_view.hash());
  init_launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                         0 /*projection id*/,
                                                         READ_ONLY,
                                                         EXCLUSIVE,
                                                         inputs[0]->region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                         0 /*projection id*/,
                                                         WRITE_ONLY,
                                                         EXCLUSIVE,
                                                         outputs[0]->region));
  init_launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *Dropout::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Dropout *dropout = (Dropout *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  assert(input_domain == output_domain);
  DropoutMeta *m = new DropoutMeta(handle, dropout, gpu_mem, output_domain);
  return m;
}

void Dropout::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(DROPOUT_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Dropout::forward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  // float alpha = 1.0f, beta = 0.0f;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  // const Dropout* dropout = (const Dropout*) task->args;
  DropoutMeta *m = *((DropoutMeta **)task->local_args);
  float const *input_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float *output_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);

  forward_kernel_wrapper(m, input_ptr, output_ptr);
}

void Dropout::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(DROPOUT_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I/O): input_grad
  regions[1](I): output_grad
*/
void Dropout::backward_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  // float alpha = 1.0f, beta = 0.0f;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  // const Dropout* dropout = (const Dropout*) task->args;
  DropoutMeta *m = *((DropoutMeta **)task->local_args);
  float *input_grad_ptr = helperGetTensorPointerRW<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float const *output_grad_ptr = helperGetTensorPointerRO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);

  backward_kernel_wrapper(m, output_grad_ptr, input_grad_ptr);
}

void Dropout::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->rate);
  sez.serialize(this->seed);
}

Node Dropout::deserialize(FFModel &ff,
                          Legion::Deserializer &dez,
                          ParallelTensor inputs[],
                          int num_inputs) {
  assert(num_inputs == 1);
  unsigned long long seed;
  float rate;
  dez.deserialize(rate);
  dez.deserialize(seed);
  DropoutParams params;
  params.rate = rate;
  params.seed = seed;
  return ff.get_or_create_node<Dropout>(inputs[0], params);
}

bool Dropout::measure_operator_cost(Simulator *sim,
                                    MachineView const &mv,
                                    CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_input, sub_output;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  assert(sub_input.get_domain() == sub_output.get_domain());
  DropoutMeta *m =
      new DropoutMeta(sim->handler, this, sim->memory, sub_output.get_domain());

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  assert(m->profiling == false);

  std::function<void()> forward, backward;
  forward = [&] { forward_kernel_wrapper(m, input_ptr, output_ptr); };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr =
        (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(input_grad_ptr != NULL);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *output_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(output_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    backward = [=] {
      backward_kernel_wrapper(m, output_grad_ptr, input_grad_ptr);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf(
        "[Measure Dropout] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Dropout] name(%s) forward_time(%.4lf)\n",
           name,
           cost_metrics.forward_time);
  }
  // Free dropoutmeta
  delete m;
  return true;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::DropoutParams>::operator()(
    FlexFlow::DropoutParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.rate);
  hash_combine(key, params.seed);
  return key;
}
}; // namespace std
