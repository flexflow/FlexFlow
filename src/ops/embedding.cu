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

Tensor FFModel::embedding(const std::string& pcname,
                          const Tensor& input,
                          int num_entries,
                          int out_dim,
                          AggrMode aggr,
                          Initializer* kernel_initializer)
{
  //assert(config.strategies.find(name) != config.strategies.end());
  //ParallelConfig pc = config.strategies[name];
  //IndexSpaceT<2> task_is = IndexSpaceT<2>(get_or_create_task_is(pc));
  Embedding* embed = new Embedding(*this, pcname, input, num_entries,
                                   out_dim, aggr, kernel_initializer);
  embed->add_to_model(*this);
  return embed->outputs[0];
}

Embedding* FFModel::embedding(const std::string& pcname,
                              int num_entries,
                              int out_dim,
                              AggrMode aggr,
                              Initializer* kernel_initializer)
{
  //assert(config.strategies.find(name) != config.strategies.end());
  //ParallelConfig pc = config.strategies[name];
  //IndexSpaceT<2> task_is = IndexSpaceT<2>(get_or_create_task_is(pc));
  Embedding* embed = new Embedding(*this, pcname, num_entries,
                                   out_dim, aggr, kernel_initializer);
  return embed;
}

Embedding::Embedding(FFModel& model,
                     const std::string& pcname,
                     const Tensor& _input,
                     //std::stirng name,
                     int num_entries, int outDim,
                     AggrMode _aggr,
                     Initializer* kernel_initializer)
: Op(pcname, _input), out_channels(outDim), aggr(_aggr), profiling(model.config.profiling)
{
  assert(_input.numDim == 2);
  create_kernel(model, num_entries, kernel_initializer);
  create_output_and_partition(model);
#ifdef DEADCODE
  // Create kernel tensor
  Rect<2> kernel_rect(Point<2>(0, 0), Point<2>(outDim-1, inDim-1));
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(float), FID_DATA);
  IndexSpaceT<2> kernel_is = runtime->create_index_space(ctx, kernel_rect);
  kernel.region = runtime->create_logical_region(ctx, kernel_is, fs);
  {
    int num_part_c = part_rect.hi[0] - part_rect.lo[0] + 1;
    int extent_c = (outDim + num_part_c - 1) / num_part_c;
    Rect<2> extent(Point<2>(0, 0), Point<2>(extent_c, inDim-1));
    Transform<2, 2> transform;
    transform[0][0] = extent_c; transform[0][1] = 0;
    transform[1][0] = 0; transform[1][1] = 0;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, kernel_is, task_is, transform, extent);
    kernel.part = runtime->get_logical_partition(
        ctx, kernel.region, ip);
  }
  // Create kernel tensor gradients
  Rect<3> kernel_grad_rect(Point<3>(0, 0, 0),
      Point<3>(outDim-1, inDim-1, part_rect.hi[1] - part_rect.lo[1]));
  IndexSpaceT<3> kernel_grad_is = runtime->create_index_space(
      ctx, kernel_grad_rect);
  kernel.region_grad = runtime->create_logical_region(
      ctx, kernel_grad_is, fs);
  {
    int num_part_c = part_rect.hi[0] - part_rect.lo[0] + 1;
    int extent_c = (outDim + num_part_c - 1) / num_part_c;
    Rect<3> extent(Point<3>(0, 0, 0), Point<3>(extent_c, inDim-1, 0));
    Transform<3, 2> transform;
    transform[0][0] = extent_c; transform[0][1] = 0;
    transform[1][0] = 0; transform[1][1] = 0;
    transform[2][0] = 0; transform[2][1] = 1;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, kernel_grad_is, task_is, transform, extent);
    kernel.part_grad = runtime->get_logical_partition(
        ctx, kernel.region_grad, ip);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    assert(runtime->is_index_partition_complete(ctx, ip));
  }
#endif
}

Embedding::Embedding(FFModel& model,
                     const std::string& pcname,
                     int num_entries, int outDim,
                     AggrMode _aggr,
                     Initializer* kernel_initializer)
: Op(pcname, 1), out_channels(outDim), aggr(_aggr), profiling(model.config.profiling)
{
  create_kernel(model, num_entries, kernel_initializer);
}

Tensor Embedding::init_inout(FFModel& model, const Tensor& _input)
{
  add_to_model(model);
  assert(_input.numDim == 2);
  inputs[0] = _input;
  create_output_and_partition(model);
  return outputs[0];
}

void Embedding::add_to_model(FFModel& model)
{
  model.layers.push_back(this);
  model.parameters.push_back(weights[0]);
}

void Embedding::create_kernel(FFModel& model, int num_entries, Initializer* kernel_initializer)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  {
    const int dims[2] = {out_channels, num_entries};
    // Embeddding weights and linear weights can be partitioned in the same way
    weights[numWeights++] = model.create_linear_weight<2>(this, dims, (IndexSpaceT<2>)task_is, DT_FLOAT, kernel_initializer);
  }
}

void Embedding::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // Currently assume we can only partition over the sample dim
  assert(part_rect.hi[0] == part_rect.lo[0]);
  {
    const int dims[2] = {inputs[0].adim[1], out_channels};
    outputs[0] = model.create_tensor<2>(dims, (IndexSpaceT<2>)task_is, DT_FLOAT);
  }
  // Compute partition bound for input
  Rect<2> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0].part;
    input_grad_lps[0] = inputs[0].part_grad;
  } else {
    // Currently assert input must have the same partition
    // to avoid data movement
    assert(false);
  }
}

//__host__
//OpMeta* Embedding::init_task(const Task *task,
//                             const std::vector<PhysicalRegion> &regions,
//                             Context ctx, Runtime* runtime)
//{}

void Embedding::init(const FFModel& ff)
{}

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
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const Embedding* embed = (Embedding*) task->args;
  TensorAccessorR<int64_t, 2> accInput(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> accOutput(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, false/*readOutput*/);
  TensorAccessorR<float, 2> accWeight(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  // Input matches Output
  assert(accInput.rect.hi[1] == accOutput.rect.hi[1]);
  assert(accInput.rect.lo[1] == accOutput.rect.lo[1]);
  // Weight matches Output
  assert(accWeight.rect.hi[1] == accOutput.rect.hi[0]);
  assert(accWeight.rect.lo[1] == accOutput.rect.lo[0]);
  int in_dim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;
  int out_dim = accOutput.rect.hi[0] - accOutput.rect.lo[0] + 1;
  int batch_size = accOutput.rect.hi[1] - accOutput.rect.lo[1] + 1;
  embed_forward<<<GET_BLOCKS(accOutput.rect.volume()), CUDA_NUM_THREADS>>>(
      accInput.ptr, accOutput.ptr, accWeight.ptr, out_dim, in_dim, batch_size, embed->aggr);
  checkCUDA(cudaDeviceSynchronize());
  if (embed->profiling) {
    print_tensor<2, int64_t>(accInput.ptr, accInput.rect, "[Embedding:forward:input]");
    print_tensor<2, float>(accWeight.ptr, accWeight.rect, "[Embedding:forward:weight]");
    print_tensor<2, float>(accOutput.ptr, accOutput.rect, "[Embedding:forward:output]");
    checkCUDA(cudaDeviceSynchronize());
  }
}

void Embedding::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(EMBED_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Embedding)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0]: input
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // regions[1]: output
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: weight
  launcher.add_region_requirement(
      RegionRequirement(weights[0].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, weights[0].region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Embedding::backward_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const Embedding* embed = (Embedding*) task->args;
  TensorAccessorR<int64_t, 2> accInput(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 2> accOutput(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> accWeightGrad(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, true/*readOutput*/);
  // Input matches Output
  assert(accInput.rect.hi[1] == accOutput.rect.hi[1]);
  assert(accInput.rect.lo[1] == accOutput.rect.lo[1]);
  // WeightGrad matches Output
  assert(accWeightGrad.rect.hi[1] - accWeightGrad.rect.lo[1] == accOutput.rect.hi[0] - accOutput.rect.lo[0]);
  int in_dim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;
  int out_dim = accOutput.rect.hi[0] - accOutput.rect.lo[0] + 1;
  int batch_size = accOutput.rect.hi[1] - accOutput.rect.lo[1] + 1;
  // Explicitly initialize accWegihtGrad to zero to aviod calling zero_gradients() before backward()
  // as an optimization for DLRM
  //assign_kernel<<<GET_BLOCKS(accWeightGrad.rect.volume()), CUDA_NUM_THREADS>>>(
  //      accWeightGrad.ptr, accWeightGrad.rect.volume(), 0.0f);
  embed_backward<<<GET_BLOCKS(accOutput.rect.volume()), CUDA_NUM_THREADS>>>(
      accInput.ptr, accOutput.ptr, accWeightGrad.ptr, out_dim, in_dim, batch_size, embed->aggr);
  checkCUDA(cudaDeviceSynchronize());
  if (embed->profiling) {
    print_tensor<2, float>(accOutput.ptr, accOutput.rect, "[Embedding:backward:output_grad]");
    print_tensor<2, float>(accWeightGrad.ptr, accWeightGrad.rect, "[Embedding:backward:weight_grad]");
    print_tensor<2, int64_t>(accInput.ptr, accInput.rect, "[Embedding:backward:input]");
    checkCUDA(cudaDeviceSynchronize());
  }
}

void Embedding::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(EMBED_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Embedding)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0]: input
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // regions[1]: output_grad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part_grad, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, outputs[0].region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: weight_grad
  launcher.add_region_requirement(
      RegionRequirement(weights[0].part_grad, 0/*projection*/,
                        READ_WRITE, EXCLUSIVE, weights[0].region_grad));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
__host__
Parameter* Embedding::get_parameter(int index)
{
  if (index == 0) {
    return &weights[0];
  } else {
    assert(0);
    return NULL;
  }
}
*/
