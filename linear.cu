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
#include "cnn_helper.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>

Tensor CnnModel::add_linear_layer(Tensor input, int output_channels, bool relu)
{
  assert(input.numDim == 2);
  Linear *li = new Linear(config, input, fc_part_is, output_channels, relu);
  layers.push_back(li);
  return li->output;
}

Linear::Linear(CnnConfig config, Tensor input, IndexSpaceT<2> part_is,
               int output_channels, bool _relu)
: Op(input), relu(_relu)
{
  assert(input.numDim == 2);
  Context ctx = config.lg_ctx;
  HighLevelRuntime* runtime = config.lg_hlr;

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }

  Rect<2, coord_t> output_rect(Point<2>(0, 0), Point<2>(output_channels-1, input.adim[1]-1));
  IndexSpaceT<2> output_is = runtime->create_index_space(ctx, output_rect);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  Transform<2, 2, coord_t> transform;
  int extent_c = (output_channels + config.fc_num_par_c - 1) / config.fc_num_par_c;
  int extent_n = (input.adim[1] + config.fc_num_par_n - 1) / config.fc_num_par_n;
  Rect<2, coord_t> extent(Point<2>(0, 0), Point<2>(extent_c-1, extent_n-1));
  transform[0][0] = extent_c; transform[0][1] = 0;
  transform[1][0] = 0; transform[1][1] = extent_n;
  IndexPartition output_ip =
    runtime->create_partition_by_restriction(ctx, output_is, part_is, transform, extent);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  
  int input_channels = input.adim[0];
  Rect<2, coord_t> kernel_rect(Point<2>(0, 0), Point<2>(output_channels * input_channels-1, config.fc_num_par_n-1));
  IndexSpaceT<2> kernel_is = runtime->create_index_space(ctx, kernel_rect);
  LogicalRegion kernel_lr = runtime->create_logical_region(ctx, kernel_is, fs);
  transform[0][0] = extent_c * input_channels;
  transform[1][1] = 1;
  Rect<2, coord_t> extent_k(Point<2>(0, 0), Point<2>(extent_c*input_channels-1, 0));
  printf("extent_k(%dx%d %d)\n", extent_c, input_channels, 1);
  IndexPartition kernel_ip =
    runtime->create_partition_by_restriction(ctx, kernel_is, part_is, transform, extent_k);
  LogicalPartition kernel_lp = runtime->get_logical_partition(ctx, kernel_lr, kernel_ip);
  TensorWithGrad kernel_tensor;
  kernel_tensor.region = kernel_lr;
  kernel_tensor.partition = kernel_lp;
  locals[0] = kernel_tensor;

  Rect<2, coord_t> bias_rect(Point<2>(0, 0), Point<2>(output_channels-1, config.fc_num_par_n-1));
  IndexSpaceT<2> bias_is = runtime->create_index_space(ctx, bias_rect);
  LogicalRegion bias_lr = runtime->create_logical_region(ctx, bias_is, fs);
  transform[0][0] = extent_c;
  transform[1][1] = 1;
  Rect<2, coord_t> extent_b(Point<2>(0, 0), Point<2>(extent_c-1,0));
  IndexPartition bias_ip =
    runtime->create_partition_by_restriction(ctx, bias_is, part_is, transform, extent_b);
  LogicalPartition bias_lp = runtime->get_logical_partition(ctx, bias_lr, bias_ip);
  TensorWithGrad bias_tensor;
  bias_tensor.region = bias_lr;
  bias_tensor.partition = bias_lp;
  locals[1] = bias_tensor;

  output.numDim = 2;
  output.adim[0] = output_channels;
  output.adim[1] = input.adim[1];
  output.pdim[0] = extent_c;
  output.pdim[1] = extent_n;
  output.region = output_lr;
  output.partition = output_lp;

  // Every partition reads all input_channels
  transform[0][0] = 0;
  transform[1][1] = extent_n;
  Rect<2, coord_t> extent_i(Point<2>(0, 0), Point<2>(input_channels-1, extent_n-1));
  IndexSpaceT<2> input_is = IndexSpaceT<2>(inputs[0].region.get_index_space());
  IndexPartition input_ip 
     = runtime->create_partition_by_restriction(ctx, input_is, part_is, transform, extent_i);
  input_lps[0] = runtime->get_logical_partition(ctx, inputs[0].region, input_ip);
}

/*
  regions[0]: input
  regions[1]: output
  regions[2]: kernel
  regions[3]: bias
*/
OpMeta* Linear::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  const int BLKSIZE = 512;
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const Linear* linear = (Linear*) task->args;
  CnnHandle handle = *((const CnnHandle*) task->local_args);
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorRO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorWO<float, 2> acc_kernel(regions[2], FID_DATA);
  const AccessorWO<float, 2> acc_bias(regions[3], FID_DATA);
  Rect<2> rect_input, rect_output, rect_kernel, rect_bias;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_kernel = runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  assert(acc_kernel.accessor.is_dense_arbitrary(rect_kernel));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  float* kernel_ptr = acc_kernel.ptr(rect_kernel.lo);
  float* bias_ptr = acc_bias.ptr(rect_bias.lo);
  curandGenerator_t genGPU;
  curandCreateGenerator(&genGPU, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(genGPU, 1234ULL);
  int input_channels = rect_input.hi[0] - rect_input.lo[0] + 1;
  int output_channels = rect_output.hi[0] - rect_output.lo[0] + 1;
  int batch_size = linear->output.pdim[1];
  printf("init linear (input): in_c(%d) out_c(%d) batch_size(%d)\n", input_channels, output_channels, batch_size);

  coord_t kernel_elements = input_channels * linear->output.pdim[0];
  float factor = 1.0f / sqrt(input_channels);
  assert(kernel_elements == rect_kernel.volume());
  curandGenerateUniform(genGPU, kernel_ptr, kernel_elements);
  int num_blocks = (kernel_elements + BLKSIZE - 1) / BLKSIZE;
  scale_kernel<<<num_blocks, BLKSIZE>>>(kernel_ptr, kernel_elements, -factor, factor);
  curandGenerateUniform(genGPU, bias_ptr, linear->output.pdim[0]);
  num_blocks = (linear->output.pdim[0] + BLKSIZE - 1) / BLKSIZE;
  scale_kernel<<<num_blocks, BLKSIZE>>>(bias_ptr, linear->output.pdim[0], -factor, factor);
  curandDestroyGenerator(genGPU);

  LinearMeta* m = new LinearMeta(handle);
  m->relu = linear->relu;
  m->input_channels = input_channels;
  m->output_channels = output_channels;
  m->batch_size = batch_size;
  float* dram_one_ptr = (float *) malloc(sizeof(float) * batch_size);
  for (int i = 0; i < batch_size; i++)
    dram_one_ptr[i] = 1.0f;
  checkCUDA(cudaMalloc(&m->one_ptr, sizeof(float) * batch_size));
  checkCUDA(cudaMemcpy(m->one_ptr, dram_one_ptr,
                       sizeof(float) * batch_size, cudaMemcpyDeviceToDevice));
  if (m->relu) {
    checkCUDNN(cudnnCreateActivationDescriptor(&m->actiDesc));
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
    checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          batch_size, output_channels, 1, 1));
  }
  return m;
}

void Linear::init(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, model.fc_part_is);
  int idx;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    CnnHandle handle = model.cnn_handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(CnnHandle)));
  }
  IndexLauncher init_launcher(LINEAR_INIT_TASK_ID, model.fc_part_is,
                              TaskArgument(this, sizeof(Linear)), argmap);
  init_launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(output.partition, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  init_launcher.add_field(1, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(locals[0].partition, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, locals[0].region));
  init_launcher.add_field(2, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(locals[1].partition, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, locals[1].region));
  init_launcher.add_field(3, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*
  regions[0](I); input
  regions[1](O): output
  regions[2](I): kernel
  regions[3](I): bias
*/
__host__
void Linear::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f, beta = 0.0f;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  int input_channels = m->input_channels;
  int output_channels = m->output_channels;
  int batch_size = m->batch_size;
  const float *one_ptr = m->one_ptr;
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRO<float, 2> acc_kernel(regions[2], FID_DATA);
  const AccessorRO<float, 2> acc_bias(regions[3], FID_DATA);
  Rect<2> rect_input, rect_output, rect_kernel, rect_bias;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_kernel = runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  // make sure the sizes match
  assert(rect_input.volume() == input_channels * batch_size);
  assert(rect_output.volume() == output_channels * batch_size);
  assert(rect_kernel.volume() == input_channels * output_channels);
  assert(rect_bias.volume() == output_channels);
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_kernel.accessor.is_dense_arbitrary(rect_kernel));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);
  const float *kernel_ptr = acc_kernel.ptr(rect_kernel.lo);
  const float *bias_ptr = acc_bias.ptr(rect_bias.lo);

  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        output_channels, batch_size, input_channels,
                        &alpha, kernel_ptr, input_channels,
                        input_ptr, input_channels, &beta,
                        output_ptr, output_channels));
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        output_channels, batch_size, 1,
                        &alpha, bias_ptr, 1,
                        one_ptr, 1, &alpha,
                        output_ptr, output_channels));
  if (m->relu) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
                                      &alpha, m->outputTensor, output_ptr,
                                      &beta, m->outputTensor, output_ptr));
  }
}

void Linear::forward(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, model.fc_part_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(LINEAR_FWD_TASK_ID, model.fc_part_is,
                         TaskArgument(NULL, 0), argmap);
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.partition, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(locals[0].partition, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, locals[0].region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(locals[1].partition, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, locals[1].region));
  launcher.add_field(3, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I/O): input
  regions[1](I): output
  regions[2](I/O): filter
  regions[3](I/O): bias
*/
__host__
void Linear::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
}

void Linear::backward(const CnnModel& model)
{
}

