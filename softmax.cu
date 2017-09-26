/* Copyright 2017 Stanford, NVIDIA
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

#include "ops.h"
#include <cuda_runtime.h>
#include <cudnn.h>

Tensor CnnModel::add_softmax_layer(Tensor input)
{
  assert(input.numDim == 2);
  Softmax *sm = new Softmax(config, input, sm_part_is);
  layers.push_back(sm);
  return sm->output;
}

Softmax::Softmax(CnnConfig config, Tensor input, IndexSpaceT<1> part_is)
: Op(input)
{
  assert(input.numDim == 2);
  Context ctx = config.lg_ctx;
  HighLevelRuntime* runtime = config.lg_hlr;

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }

  IndexSpaceT<2> output_is(input.region.get_index_space());
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  Transform<2, 1, coord_t> transform;
  int extent_c = input.adim[0];
  int extent_n = (input.adim[1] + config.sm_num_par - 1) / config.sm_num_par;
  Rect<2, coord_t> extent(Point<2>(0, 0), Point<2>(extent_c-1, extent_n-1));
  transform[0][0] = 0;
  transform[1][0] = extent_n;
  IndexPartition output_ip
    = runtime->create_partition_by_restriction(ctx, output_is, part_is, transform, extent);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  output.numDim = 2;
  output.adim[0] = input.adim[0];
  output.adim[1] = input.adim[1];
  output.pdim[0] = extent_c;
  output.pdim[1] = extent_n;
  output.region = output_lr;
  output.partition = output_lp;    
  // Every partition reads all input_channels
  // Use the same partitioning as outputs
  IndexPartition input_ip = output_ip;
  input_lps[0] = runtime->get_logical_partition(ctx, inputs[0].region, input_ip);
}

/*
  regions[0]: input
  regions[1]: output
 */
OpMeta* Softmax::init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Softmax* softmax = (Softmax*) task->args;
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  Rect<2> rect_input, rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  CnnHandle handle = *((const CnnHandle*) task->local_args);
  SoftmaxMeta* m = new SoftmaxMeta(handle);
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  //checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  assert(rect_input == rect_output);
  int input_c = rect_input.hi[0] - rect_input.lo[0] + 1;
  int input_n = rect_input.hi[1] - rect_input.lo[1] + 1;
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n, input_c, 1, 1));
  return m;
}

__host__
void Softmax::init(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<1> rect = runtime->get_index_space_domain(ctx, model.sm_part_is);
  int idx = 0;
  for (PointInRectIterator<1> it(rect); it(); it++) {
    CnnHandle handle = model.cnn_handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(CnnHandle)));
  }
  IndexLauncher init_launcher(SOFTMAX_INIT_TASK_ID, model.sm_part_is,
                              TaskArgument(this, sizeof(Softmax)), argmap);
  init_launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(output.partition, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  init_launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<1> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/
__host__
void Softmax::forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  float alpha = 1.0f, beta = 0.0f;
  const SoftmaxMeta* m = *((SoftmaxMeta**) task->local_args);
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  Rect<2> rect_input, rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);

  checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha, m->inputTensor, input_ptr,
                                 &beta, m->inputTensor, output_ptr));
}

__host__
void Softmax::forward(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<1> rect = runtime->get_index_space_domain(ctx, model.sm_part_is);
  int idx = 0;
  for (PointInRectIterator<1> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(SOFTMAX_FWD_TASK_ID, model.sm_part_is,
                         TaskArgument(NULL, 0), argmap);
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.partition, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

__global__ void SoftmaxLossBackprop(float *input, const int *label, int num_labels, int batch_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;
  int label_idx = label[idx];
  input[idx * num_labels + label_idx] -= 1.0f;
}

/*
  regions[0](I/O): input
  regions[1](I): output
  regions[2](I): labels
*/
__host__
void Softmax::backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
  const int BLK_SIZE = 512;
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  float alpha = 1.0f, beta = 0.0f;
  const SoftmaxMeta* m = *((SoftmaxMeta**) task->local_args);
  const AccessorRW<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorRO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRO<int, 1> acc_label(regions[2], FID_DATA);
  Rect<2> rect_input, rect_output;
  Rect<1> rect_label;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_label = runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  // make sure the image indices match!
  assert(rect_label.hi[0] == rect_output.hi[1]);
  assert(rect_label.lo[0] == rect_output.lo[1]);
  int num_images = rect_label.volume();
  int num_labels = rect_output.hi[0] - rect_output.lo[0] + 1;
  assert(num_labels == 1000); // check that we have 1000 different labels

  float *input_ptr = acc_input.ptr(rect_input.lo);
  const float *output_ptr = acc_output.ptr(rect_output.lo);
  const int *label_ptr = acc_label.ptr(rect_label.lo);
  checkCUDA(cudaMemcpy(input_ptr, output_ptr,
                       rect_input.volume() * sizeof(float),
                       cudaMemcpyDeviceToDevice));
  int num_blocks = (num_images + BLK_SIZE - 1) / BLK_SIZE;
  SoftmaxLossBackprop<<<num_blocks, BLK_SIZE>>>(input_ptr, label_ptr, num_labels, num_images);

  // Accouting for batch size in SGD
  float scalVal = 1.0f / static_cast<float>(num_images);
  checkCUDA(cublasSscal(m->handle.blas, rect_input.volume(), &scalVal, input_ptr, 1));
}

__host__
void Softmax::backward(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  int idx = 0;
  Rect<1> rect = runtime->get_index_space_domain(ctx, model.sm_part_is);
  for (PointInRectIterator<1> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(SOFTMAX_BWD_TASK_ID, model.sm_part_is,
                         TaskArgument(NULL, 0), argmap);
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.partition, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(model.input_label.partition, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, model.input_label.region));
  launcher.add_field(2, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}
