/* Copyright 2020 Stanford
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
#include "cuda_helper.h"

using namespace Legion;

Tensor FFModel::embedding(const Tensor input,
                          int num_entries,
                          int out_dim,
                          AggrMode aggr,
                          const Op* shared_op,
                          Initializer* kernel_initializer,
                          const char* name)
{
  {
    Embedding* embed = new Embedding(*this, input, num_entries, out_dim,
                                     aggr, name);
    layers.push_back(embed);
    return embed->outputs[0];
  }
#ifdef OLD_LAYER_CREATION
  if (kernel_initializer == NULL) {
    int seed = std::rand();
    kernel_initializer = new GlorotUniform(seed);
  }
#ifdef FF_USE_NCCL
  ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
  ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
  Tensor weight;
  {
    const int dims[2] = {num_entries, out_dim};
    weight = create_weight<2>(dims, DT_FLOAT, NULL/*owner_op*/,
        true/*create_grad*/, kernel_initializer, comm_type);
  }
  Embedding* embed = new Embedding(*this, input, weight, aggr, name);
  layers.push_back(embed);
  return embed->outputs[0];
#endif
}

Embedding::Embedding(FFModel& model,
                     const Tensor _input,
                     int _num_entries,
                     int _out_channels,
                     AggrMode _aggr,
                     const char* name)
: Op(model, OP_EMBEDDING, name, 1/*inputs*/, 0/*weighs*/, _input),
  num_entries(_num_entries), out_channels(_out_channels), aggr(_aggr)
{
  numOutputs = 1;
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 1; i < numdim; i++) {
    dims[i] = _input->dims[i];
    assert(i == 1 || dims[i].degree == 1);
  }
  dims[0].size = out_channels;
  dims[0].degree = 1;
  dims[0].parallel_idx = -1;
  outputs[0] = model.create_tensor_legion_ordering(
      numdim, dims, DT_FLOAT, this);
  for (int i = 1; i < numdim; i++) {
    register_output_input_parallel_dims(outputs[0], i, inputs[0], i);
  }
  // Check correctness
  assert(check_output_input_weight_parallel_dims());
}

Embedding::Embedding(FFModel& model,
                     const Tensor _input,
                     const Tensor _weight,
                     AggrMode _aggr,
                     const char* name)
: Op(model, OP_EMBEDDING, name, 1/*inputs*/, 1/*weights*/, _input, _weight),
  num_entries(_weight->dims[1].size), out_channels(_weight->dims[0].size), aggr(_aggr)
{
  numOutputs = 1;
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 1; i < numdim; i++) {
    dims[i] = _input->dims[i];
    assert(i == 1 || dims[i].degree == 1);
  }
  dims[0].size = out_channels;
  // Assert no parallelism along the first dim
  assert(inputs[0]->dims[0].degree == 1);
  assert(inputs[0]->dims[0].parallel_idx == -1);
  outputs[0] = model.create_tensor_legion_ordering(numdim, dims, DT_FLOAT, this);
  for (int i = 1; i < numdim; i++) {
    register_output_input_parallel_dims(outputs[0], i, inputs[0], i);
  }
  // sample dim and replica dim
  register_output_weight_parallel_dims(outputs[0], 1, weights[0], 2);
  // Check correctness
  assert(check_output_input_weight_parallel_dims());
}

#ifdef DEADCODE
void Embedding::create_weights(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
#ifdef FF_USE_NCCL
  ParameterSyncType comm_type = ParameterSyncType::NCCL;  
#else
  ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
  {
    const int dims[2] = {out_channels, num_entries};
    // Embeddding weights and linear weights can be partitioned in the same way
    weights[0] = model.create_linear_weight<2, 2>(this, dims, DT_FLOAT, kernel_initializer, true/*create_grad*/, comm_type);
    assert(numWeights == 1);
  }
}
#endif

#ifdef DEADCODE
void Embedding::create_input_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // Currently assume we can only partition over the sample dim
  assert(part_rect.hi[0] == part_rect.lo[0]);
  return Op::create_input_partition(model);
#ifdef DEADCODE
  {
    const int dims[2] = {inputs[0].adim[1], out_channels};
    outputs[0] = model.create_tensor<2>(dims, DT_FLOAT, this);
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }
  // Compute partition bound for input
  Rect<2> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0]->part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0]->part;
    input_grad_lps[0] = inputs[0]->part_grad;
  } else {
    model.create_disjoint_partition<2>(
      inputs[0], (IndexSpaceT<2>)task_is, input_lps[0], input_grad_lps[0]);
  }
#endif
}
#endif

__host__
OpMeta* Embedding::init_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime* runtime)
{
  const Embedding* embed = (Embedding*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  EmbeddingMeta* m = new EmbeddingMeta(handle);
  m->profiling = embed->profiling;
  m->aggr = embed->aggr;
  return m;
}

void Embedding::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(EMBED_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Embedding)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0]: input
  //launcher.add_region_requirement(
  //  RegionRequirement(input_lps[0], 0/*projection*/,
  //    READ_ONLY, EXCLUSIVE, inputs[0]->region));
  //launcher.add_field(0, FID_DATA);
  // regions[1]: output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[2]: weight
  launcher.add_region_requirement(
    RegionRequirement(weights[0]->part, 0/*projection*/,
      READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(1, FID_DATA);
  // regions[3]: input_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection*/,
      WRITE_ONLY, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

__global__
void embed_forward(const int64_t* input,
                   float* output,
                   const float* embed,
                   int out_dim,
                   int in_dim,
                   int batch_size,
                   AggrMode aggr)
{
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    for (int j = 0; j < in_dim; j++) {
      int64_t wordIdx = input[idx * in_dim + j];
      output[i] += embed[wordIdx * out_dim + off];
      if (aggr == AGGR_MODE_SUM) {
      } else {
        assert(aggr == AGGR_MODE_AVG);
        output[i] /= in_dim;
      }
    }
  }
}

__global__
void embed_backward(const int64_t* input,
                    const float* output,
                    float* embed,
                    int out_dim,
                    int in_dim,
                    int batch_size,
                    AggrMode aggr)
{
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    int idx = i / out_dim;
    int off = i % out_dim;
    float gradient;
    if (aggr == AGGR_MODE_SUM) {
       gradient = output[i];
    } else {
      assert(aggr == AGGR_MODE_AVG);
      gradient = output[i] / in_dim;
    }
    for (int j = 0; j < in_dim; j++) {
      int64_t wordIdx = input[idx * in_dim + j];
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
    }
  }
}

void Embedding::forward_kernel(int64_t const *input_ptr,
                               float *output_ptr,
                               float const *weight_ptr,
                               int in_dim,
                               int out_dim,
                               int batch_size,
                               AggrMode aggr,
                               int outputSize)
{
  embed_forward<<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS>>>(
      input_ptr, output_ptr, weight_ptr, out_dim, in_dim, batch_size, aggr);
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I): kernel
*/
__host__
void Embedding::forward_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime* runtime)
{
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (in_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return forward_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int NDIM>
__host__
void Embedding::forward_task_with_dim(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  //const Embedding* embed = (Embedding*) task->args;
  const EmbeddingMeta* m = *((EmbeddingMeta**) task->local_args);
  TensorAccessorR<int64_t, NDIM> accInput(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, NDIM> accOutput(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, false/*readOutput*/);
  TensorAccessorR<float, NDIM> accWeight(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  // Input matches Output
  for (int i = 1; i < NDIM; i++) {
    assert(accInput.rect.hi[i] == accOutput.rect.hi[i]);
    assert(accInput.rect.lo[i] == accOutput.rect.lo[i]);
  }
  // Weight matches Output
  assert(accWeight.rect.hi[0] - accWeight.rect.lo[0]
      == accOutput.rect.hi[0] - accOutput.rect.lo[0]);
  int in_dim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;
  int out_dim = accOutput.rect.hi[0] - accOutput.rect.lo[0] + 1;
  int batch_size = accOutput.rect.volume() / out_dim;
  forward_kernel(accInput.ptr, accOutput.ptr, accWeight.ptr, in_dim, out_dim, batch_size,  m->aggr, accOutput.rect.volume());
  if (m->profiling) {
    checkCUDA(cudaDeviceSynchronize());
    print_tensor<int64_t>(accInput.ptr, accInput.rect.volume(), "[Embedding:forward:input]");
    print_tensor<float>(accWeight.ptr, accWeight.rect.volume(), "[Embedding:forward:weight]");
    print_tensor<float>(accOutput.ptr, accOutput.rect.volume(), "[Embedding:forward:output]");
  }
}

void Embedding::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(EMBED_FWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0]: input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1]: output
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: weight
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Embedding::backward_kernel(int64_t const *input_ptr,
                                float const *output_ptr,
                                float *weight_grad_ptr,
                                int in_dim,
                                int out_dim,
                                int batch_size,
                                AggrMode aggr,
                                int outputSize)
{
  embed_backward<<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS>>>(
      input_ptr, output_ptr, weight_grad_ptr, out_dim, in_dim, batch_size, aggr);
}

void Embedding::backward_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (in_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return backward_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int NDIM>
__host__
void Embedding::backward_task_with_dim(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  //const Embedding* embed = (Embedding*) task->args;
  const EmbeddingMeta* m = *((EmbeddingMeta**) task->local_args);
  TensorAccessorR<int64_t, NDIM> accInput(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, NDIM> accOutput(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorW<float, NDIM> accWeightGrad(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, true/*readOutput*/);
  // Input matches Output
  for (int i = 1; i < NDIM; i++) {
    assert(accInput.rect.hi[i] == accOutput.rect.hi[i]);
    assert(accInput.rect.lo[i] == accOutput.rect.lo[i]);
  }
  // WeightGrad matches Output
  assert(accWeightGrad.rect.hi[0] - accWeightGrad.rect.lo[0] == accOutput.rect.hi[0] - accOutput.rect.lo[0]);
  int in_dim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;
  int out_dim = accOutput.rect.hi[0] - accOutput.rect.lo[0] + 1;
  int batch_size = accOutput.rect.volume() / out_dim;
  backward_kernel(accInput.ptr, accOutput.ptr, accWeightGrad.ptr, in_dim, out_dim, batch_size, m->aggr, accOutput.rect.volume());
  if (m->profiling) {
    checkCUDA(cudaDeviceSynchronize());
    print_tensor<float>(accOutput.ptr, accOutput.rect.volume(), "[Embedding:backward:output_grad]");
    print_tensor<float>(accWeightGrad.ptr, accWeightGrad.rect.volume(), "[Embedding:backward:weight_grad]");
    print_tensor<int64_t>(accInput.ptr, accInput.rect.volume(), "[Embedding:backward:input]");
  }
}

void Embedding::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(EMBED_BWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0]: input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1]: output_grad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: weight_grad
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part_grad, 0/*projection*/,
                        READ_WRITE, EXCLUSIVE, weights[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Embedding::measure_operator_cost(Simulator* sim,
                                      const ParallelConfig& pc,
                                      CostMetrics& cost_metrics) const
{
  TensorBase sub_input, sub_output;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input, op_type)) {
    return false;
  }

  sim->free_all();
  int64_t *input_ptr = (int64_t *)sim->allocate(sub_input.get_volume(), DT_INT64);
  assert (input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);
  float *weight_ptr = (float *)sim->allocate(num_entries * out_channels, DT_FLOAT);
  assert (weight_ptr != NULL);
  int in_dim = sub_input.dims[0].size;
  int out_dim = sub_input.dims[0].size;
  assert (sub_input.dims[1] == sub_output.dims[1]);
  int batch_size = sub_input.dims[1].size;

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(input_ptr, output_ptr, weight_ptr, in_dim, out_dim, batch_size, this->aggr, sub_output.get_volume());
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *weight_grad_ptr = (float *)sim->allocate(num_entries * out_channels, DT_FLOAT);
    assert (weight_grad_ptr != NULL);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    int64_t *input_grad_ptr = (int64_t *)sim->allocate(sub_input.get_volume(), DT_INT64);
    assert (input_grad_ptr != NULL);

    backward = [&] {
      backward_kernel(input_grad_ptr, output_grad_ptr, weight_grad_ptr, in_dim, out_dim, batch_size,
        this->aggr, sub_output.get_volume());
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Embedding] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Embedding] name(%s) forward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time);
  }

  return true;
}

Node FFModel::get_or_create_embedding_node(const Tensor input,
                                           int num_entries,
                                           int out_channels,
                                           AggrMode aggr)
{
  size_t hash = input->get_owner_independent_hash();
  hash = hash * 31 + std::hash<int>()(num_entries);
  hash = hash * 31 + std::hash<int>()(out_channels);
  hash = hash * 31 + std::hash<int>()(aggr);
  const auto& it = cached_embedding_ops.find(hash);
  Embedding* embed = NULL;
  if (it != cached_embedding_ops.end()) {
    embed = it->second;
  } else {
    embed = new Embedding(*this, input, num_entries, out_channels,
                          aggr, NULL);
    cached_embedding_ops[hash] = embed;
  }
  Node ret;
  ret.guid = node_global_guid ++;
  ret.ptr = embed;
  return ret;
}
