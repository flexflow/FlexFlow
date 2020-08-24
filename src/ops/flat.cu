/* Copyright 2018 Stanford
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

Tensor FFModel::flat(const Tensor& input)
{
  assert(input.numDim == 4);
  //assert(strategies.find(name) != strategies.end());
  //ParallelConfig pc = strategies[name];
  Flat *flat = new Flat(*this, input);
  layers.push_back(flat);
  return flat->outputs[0];
}

Flat* FFModel::flat()
{
  //assert(strategies.find(name) != strategies.end());
  //ParallelConfig pc = strategies[name];
  Flat *flat = new Flat(*this);
  layers.push_back(flat);
  return flat;
}

Flat::Flat(FFModel& model,
           const Tensor& _input)
: Op(model, OP_FLAT, "Flat", _input)
{
  assert(_input.numDim == 4);
  int out_dim = _input.adim[0] * _input.adim[1] * _input.adim[2];
  int batch_size = _input.adim[3];
  outputs[0].numDim = 2;
  outputs[0].adim[0] = out_dim;
  outputs[0].adim[1] = batch_size;
}

Flat::Flat(FFModel& model)
: Op(model, OP_FLAT, "Flat", 1)
{
}

Tensor Flat::init_inout(FFModel& model, const Tensor& _input)
{
  inputs[0] = _input;
  create_output_and_partition(model);
  return outputs[0];
}

/*
void Flat::add_to_model(FFModel& model)
{
  model.layers.push_back(this);
}
*/

void Flat::create_weights(FFModel& model)
{
  // Do nothing
}

void Flat::create_output_and_partition(FFModel& model)
{
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_c = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_n = part_rect.hi[1] - part_rect.lo[1] + 1;
  // Assert data parallelism for operators with dim changes
  assert(num_par_c == 1);
 
  int out_dim = inputs[0].adim[0] * inputs[0].adim[1] * inputs[0].adim[2];
  int batch_size = inputs[0].adim[3];
  // Create output tensor
  {
    const int dims[2] = {batch_size, out_dim};
    outputs[0] = model.create_tensor<2>(dims, (IndexSpaceT<2>)task_is, DT_FLOAT);
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }
  model.create_data_parallel_partition_with_diff_dims<4, 2>(
      inputs[0], (IndexSpaceT<2>)task_is, input_lps[0], input_grad_lps[0]);
}

OpMeta* Flat::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  FFHandler handler = *((const FFHandler*) task->local_args);
  FlatMeta* m = new FlatMeta(handler);
  return m;
}

void Flat::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }

  IndexLauncher launcher(FLAT_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Flat)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/  
void Flat::forward_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);

  assert(acc_input.rect.volume() == acc_output.rect.volume());
  checkCUDA(cudaMemcpyAsync(acc_output.ptr, acc_input.ptr,
                            acc_input.rect.volume() * sizeof(float),
                            cudaMemcpyDeviceToDevice));
  checkCUDA(cudaDeviceSynchronize());
}

void Flat::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(FLAT_FWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I/O) : input_grad
  regions[1](I) : output_grad
*/
void Flat::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  float alpha = 1.0f;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorW<float, 4> acc_input_grad(
      regions[0], task->regions[0], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, 2> acc_output_grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  assert(acc_input_grad.rect.volume() == acc_output_grad.rect.volume());
  apply_add_with_scale<<<GET_BLOCKS(acc_input_grad.rect.volume()), CUDA_NUM_THREADS>>>(
      acc_input_grad.ptr, acc_output_grad.ptr, acc_input_grad.rect.volume(), alpha);
  //checkCUDA(cudaMemcpyAsync(acc_input_grad.ptr, acc_output_grad.ptr,
  //                          acc_input_grad.rect.volume() * sizeof(float),
  //                          cudaMemcpyDeviceToDevice));
  checkCUDA(cudaDeviceSynchronize());
}

void Flat::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(FLAT_BWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Flat::measure_compute_time(Simulator* sim,
                                const ParallelConfig& pc,
                                float& forward_time,
                                float& backward_time)
{
  // Assume flat has no cost
  forward_time = 0;
  backward_time = 0;
  return true;
}

Domain Flat::get_input_tensor_shape(const ParallelConfig& pc,
                                  int input_idx, int part_idx)
{
  assert(input_idx < numInputs);
  assert(pc.nDims == 2);
  // Currently assume data parallelism for Flat
  assert(pc.dim[0] == 1);
  Domain d;
  d.dim = inputs[input_idx].numDim;
  for (int i = 0; i < d.dim-1; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+d.dim] = inputs[input_idx].adim[i] - 1;
  }
  assert(inputs[input_idx].adim[d.dim-1] % pc.num_parts() == 0);
  int dim_size = inputs[input_idx].adim[d.dim-1] / pc.num_parts();
  d.rect_data[d.dim-1] = part_idx * dim_size;
  d.rect_data[2*d.dim-1] = d.rect_data[d.dim-1] + dim_size - 1;
  return d;
}
