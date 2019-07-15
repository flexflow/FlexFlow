/* Copyright 2019 Stanford, NVIDIA
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

Tensor FFModel::dense(std::string name,
                      const Tensor& input,
                      int outDim, 
                      ActiMode activation,
                      Initializer* kernel_initializer,
                      Initializer* bias_initializer)
{
  assert(input.numDim == 2);
  assert(config.strategies.find(name) != config.strategies.end());
  ParallelConfig pc = config.strategies[name];
  IndexSpaceT<2> task_is = IndexSpaceT<2>(get_or_create_task_is(pc));
  Linear *li = new Linear(name, config, input, task_is, outDim, activation,
                          kernel_initializer, bias_initializer);
  layers.push_back(li);
  return li->output;
}

//
// Deprecated -- TO BE REMOVED
//
Tensor FFModel::linear(std::string name,
                       const Tensor& input,
                       int out_dim,
                       ActiMode activation,
                       Initializer* kernel_initializer,
                       Initializer* bias_initializer)
{
  return dense(name, input, out_dim, activation,
               kernel_initializer, bias_initializer);
}

Linear::Linear(std::string _name,
               const FFConfig& _config,
               const Tensor& _input,
               const IndexSpaceT<2>& _task_is,
               int outDim,
               ActiMode _activation,
               Initializer* kernel_initializer,
               Initializer* bias_initializer)
: Op(_name, _input), task_is(_task_is),
  activation(_activation),
  profiling(_config.profiling)
  // in_channels(_input.adim[0]), out_channels(outDim)
{
  assert(_input.numDim == 2);
  Context ctx = _config.lg_ctx;
  HighLevelRuntime* runtime = _config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);
  fc_num_par_c = part_rect.hi[0] - part_rect.lo[0] + 1;
  int fc_num_par_n = part_rect.hi[1] - part_rect.lo[1] + 1;
  num_replica = fc_num_par_n;
  int inDim = _input.adim[0];

  printf("Linear fc_num_par_c(%d) fc_num_par_n(%d)\n", fc_num_par_c, fc_num_par_n);
  FieldSpace fs = _config.field_space;

  Rect<2> output_rect(Point<2>(0, 0), Point<2>(outDim-1, _input.adim[1]-1));
  IndexSpaceT<2> output_is = runtime->create_index_space(ctx, output_rect);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  LogicalRegion output_grad_lr = runtime->create_logical_region(ctx, output_is, fs);
  Transform<2, 2, coord_t> transform;
  int extent_c = (outDim + fc_num_par_c - 1) / fc_num_par_c;
  int extent_n = (_input.adim[1] + fc_num_par_n - 1) / fc_num_par_n;
  Rect<2> extent(Point<2>(0, 0), Point<2>(extent_c-1, extent_n-1));
  transform[0][0] = extent_c; transform[0][1] = 0;
  transform[1][0] = 0; transform[1][1] = extent_n;
  IndexPartition output_ip =
    runtime->create_partition_by_restriction(ctx, output_is, task_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, output_ip));
  assert(runtime->is_index_partition_complete(ctx, output_ip));
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  LogicalPartition output_grad_lp = runtime->get_logical_partition(ctx, output_grad_lr, output_ip);

  // Note: we only need replica's grad, so no need to create lr/lp for forward
  Rect<2, coord_t> replica_rect(Point<2>(0, 0),
                       Point<2>(inDim*fc_num_par_c-1, _input.adim[1]-1));
  IndexSpaceT<2> replica_is = runtime->create_index_space(ctx, replica_rect);
  LogicalRegion replica_lr = runtime->create_logical_region(ctx, replica_is, fs);
  transform[0][0] = inDim;
  transform[1][1] = extent_n;
  Rect<2, coord_t> extent_r(Point<2>(0, 0), Point<2>(inDim-1, extent_n-1));
  IndexPartition replica_ip =
    runtime->create_partition_by_restriction(ctx, replica_is, task_is, transform, extent_r);
  assert(runtime->is_index_partition_disjoint(ctx, replica_ip));
  assert(runtime->is_index_partition_complete(ctx, replica_ip));
  LogicalPartition replica_lp = runtime->get_logical_partition(ctx, replica_lr, replica_ip);
  Tensor replica_tensor;
  replica_tensor.region = LogicalRegion::NO_REGION;
  replica_tensor.part = LogicalPartition::NO_PART;
  replica_tensor.region_grad = replica_lr;
  replica_tensor.part_grad = replica_lp;
  locals[0] = replica_tensor;
  // Create subpartitions for backward prop aggregation
  for (int i = 0; i < fc_num_par_c; i++) {
    transform[0][0] = _input.pdim[0];
    transform[1][1] = _input.pdim[1];
    Rect<2, coord_t> ext(Point<2>(inDim*i, 0),
                         Point<2>(inDim*i + _input.pdim[0] - 1,
                                  _input.pdim[1]-1));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, replica_is, task_is, transform, ext);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    replica_sub_lps[i] = runtime->get_logical_partition(ctx, replica_lr, ip);
  }

  Rect<1, coord_t> kernel_rect(0, inDim * outDim - 1);
  Rect<2, coord_t> kernel_grad_rect(Point<2>(0, 0), Point<2>(outDim * inDim-1, fc_num_par_n-1));
  IndexSpaceT<1> kernel_is = runtime->create_index_space(ctx, kernel_rect);
  IndexSpaceT<2> kernel_grad_is = runtime->create_index_space(ctx, kernel_grad_rect);
  LogicalRegion kernel_lr = runtime->create_logical_region(ctx, kernel_is, fs);
  LogicalRegion kernel_grad_lr = runtime->create_logical_region(ctx, kernel_grad_is, fs);
  transform[0][0] = extent_c * inDim;
  transform[1][1] = 1;
  Rect<2, coord_t> extent_k_grad(Point<2>(0, 0), Point<2>(extent_c*inDim-1, 0));
  printf("extent_k(%dx%d %d)\n", extent_c, inDim, 1);
  IndexPartition kernel_grad_ip =
    runtime->create_partition_by_restriction(ctx, kernel_grad_is, task_is,
                                             transform, extent_k_grad);
  assert(runtime->is_index_partition_disjoint(ctx, kernel_grad_ip));
  assert(runtime->is_index_partition_complete(ctx, kernel_grad_ip));
  LogicalPartition kernel_grad_lp =
    runtime->get_logical_partition(ctx, kernel_grad_lr, kernel_grad_ip);
  Transform<1, 2, coord_t> trans;
  trans[0][0] = extent_c * inDim; trans[0][1] = 0;
  Rect<1, coord_t> extent_k(0, extent_c*inDim-1);
  IndexPartition kernel_ip =
    runtime->create_partition_by_restriction(ctx, kernel_is, task_is, trans, extent_k);
  LogicalPartition kernel_lp =
    runtime->get_logical_partition(ctx, kernel_lr, kernel_ip);
  Tensor kernel_tensor;
  kernel_tensor.region = kernel_lr;
  kernel_tensor.part = kernel_lp;
  kernel_tensor.region_grad = kernel_grad_lr;
  kernel_tensor.part_grad = kernel_grad_lp;
  locals[1] = kernel_tensor;
  kernel_initializer->init(ctx, runtime, &kernel_tensor);

  Rect<1, coord_t> bias_rect(0, outDim-1);
  Rect<2, coord_t> bias_grad_rect(Point<2>(0, 0), Point<2>(outDim-1, fc_num_par_n-1));
  IndexSpaceT<1> bias_is = runtime->create_index_space(ctx, bias_rect);
  IndexSpaceT<2> bias_grad_is = runtime->create_index_space(ctx, bias_grad_rect);
  LogicalRegion bias_lr = runtime->create_logical_region(ctx, bias_is, fs);
  LogicalRegion bias_grad_lr = runtime->create_logical_region(ctx, bias_grad_is, fs);
  transform[0][0] = extent_c;
  transform[1][1] = 1;
  Rect<2, coord_t> extent_b_grad(Point<2>(0, 0), Point<2>(extent_c-1,0));
  IndexPartition bias_grad_ip =
    runtime->create_partition_by_restriction(ctx, bias_grad_is, task_is,
                                             transform, extent_b_grad);
  assert(runtime->is_index_partition_disjoint(ctx, bias_grad_ip));
  assert(runtime->is_index_partition_complete(ctx, bias_grad_ip));
  LogicalPartition bias_grad_lp =
    runtime->get_logical_partition(ctx, bias_grad_lr, bias_grad_ip);
  trans[0][0] = extent_c; trans[0][1] = 0;
  Rect<1, coord_t> extent_b(0, extent_c-1);
  IndexPartition bias_ip =
    runtime->create_partition_by_restriction(ctx, bias_is, task_is, trans, extent_b);
  LogicalPartition bias_lp =
    runtime->get_logical_partition(ctx, bias_lr, bias_ip);
  Tensor bias_tensor;
  bias_tensor.region = bias_lr;
  bias_tensor.part = bias_lp;
  bias_tensor.region_grad = bias_grad_lr;
  bias_tensor.part_grad = bias_grad_lp;
  locals[2] = bias_tensor;
  bias_initializer->init(ctx, runtime, &bias_tensor);
  numLocals = 3;

  output.numDim = 2;
  output.adim[0] = outDim;
  output.adim[1] = _input.adim[1];
  output.pdim[0] = extent_c;
  output.pdim[1] = extent_n;
  output.region = output_lr;
  output.part = output_lp;
  output.region_grad = output_grad_lr;
  output.part_grad = output_grad_lp;

  // Every partition reads all inDim
  transform[0][0] = 0;
  transform[1][1] = extent_n;
  Rect<2, coord_t> extent_i(Point<2>(0, 0), Point<2>(inDim-1, extent_n-1));
  IndexSpaceT<2> input_is = IndexSpaceT<2>(inputs[0].region.get_index_space());
  IndexPartition input_ip 
     = runtime->create_partition_by_restriction(ctx, input_is, task_is, transform, extent_i);
  input_lps[0] = runtime->get_logical_partition(ctx, inputs[0].region, input_ip);
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2]: replica
  regions[3](I): kernel
  regions[4](I): bias
*/
OpMeta* Linear::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 5);
  assert(task->regions.size() == 5);
  const Linear* linear = (Linear*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  //const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  //const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  //const AccessorRO<float, 1> acc_kernel(regions[3], FID_DATA);
  //const AccessorRO<float, 1> acc_bias(regions[4], FID_DATA);
  Rect<2> rect_input, rect_output, rect_replica;
  Rect<1> rect_kernel, rect_bias;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_replica = runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_kernel = runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(ctx, task->regions[4].region.get_index_space());
  assert(rect_replica.volume() == rect_input.volume());
  //assert(acc_kernel.accessor.is_dense_arbitrary(rect_kernel));
  //assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  //const float* kernel_ptr = acc_kernel.ptr(rect_kernel.lo);
  //const float* bias_ptr = acc_bias.ptr(rect_bias.lo);
  int input_channels = rect_input.hi[0] - rect_input.lo[0] + 1;
  int output_channels = rect_output.hi[0] - rect_output.lo[0] + 1;
  int batch_size = linear->output.pdim[1];
  printf("init linear (input): in_c(%d) out_c(%d) batch_size(%d)\n", input_channels, output_channels, batch_size);
  LinearMeta* m = new LinearMeta(handle);
  m->activation = linear->activation;
  m->in_channels = input_channels;
  m->out_channels = output_channels;
  m->batch_size = batch_size;

  float* dram_one_ptr = (float *) malloc(sizeof(float) * batch_size);
  for (int i = 0; i < batch_size; i++)
    dram_one_ptr[i] = 1.0f;
  checkCUDA(cudaMalloc(&m->one_ptr, sizeof(float) * batch_size));
  checkCUDA(cudaMemcpy(m->one_ptr, dram_one_ptr,
                       sizeof(float) * batch_size, cudaMemcpyHostToDevice));
  if (m->activation == AC_MODE_RELU) {
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

#ifdef DEADCODE
/*
  regions[0](O): filter
  regions[1](O): bias
*/
__host__
void Linear::init_para_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Linear* linear = (Linear*) task->args;
  const AccessorWO<float, 1> acc_filter(regions[0], FID_DATA);
  const AccessorWO<float, 1> acc_bias(regions[1], FID_DATA);
  Rect<1> rect_filter, rect_bias;
  rect_filter = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_filter.accessor.is_dense_arbitrary(rect_filter));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  float *filter_ptr = acc_filter.ptr(rect_filter.lo);
  float *bias_ptr = acc_bias.ptr(rect_bias.lo);
  // init filter and bias
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  curandGenerator_t genGPU;
  curandCreateGenerator(&genGPU, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetStream(genGPU, stream);
  curandSetPseudoRandomGeneratorSeed(genGPU, 1234ULL);
  coord_t filter_elements = linear->in_channels * linear->out_channels;
  float factor = 1.0f / sqrt(linear->in_channels);
  assert(filter_elements == rect_filter.volume());
  curandGenerateUniform(genGPU, filter_ptr, filter_elements);
  scale_kernel<<<GET_BLOCKS(filter_elements), CUDA_NUM_THREADS>>>(
      filter_ptr, filter_elements, -factor, factor);
  curandGenerateUniform(genGPU, bias_ptr, linear->out_channels);
  assert(linear->out_channels == rect_bias.volume());
  scale_kernel<<<GET_BLOCKS(linear->out_channels), CUDA_NUM_THREADS>>>(
      bias_ptr, linear->out_channels, -factor, factor);
  curandDestroyGenerator(genGPU);
}
#endif

void Linear::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;

#ifdef DEADCODE
  // First we initialize the filter and bias parameters
  {
    TaskLauncher para_launcher(LINEAR_INIT_PARA_TASK_ID, TaskArgument(this, sizeof(Linear)));
    para_launcher.add_region_requirement(
        RegionRequirement(locals[1].region, WRITE_DISCARD, EXCLUSIVE, locals[1].region));
    para_launcher.add_field(0, FID_DATA);
    para_launcher.add_region_requirement(
        RegionRequirement(locals[2].region, WRITE_DISCARD, EXCLUSIVE, locals[2].region));
    para_launcher.add_field(1, FID_DATA);
    runtime->execute_task(ctx, para_launcher);
  }
#endif

  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher init_launcher(LINEAR_INIT_TASK_ID, task_is,
                              TaskArgument(this, sizeof(Linear)), argmap);
  init_launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  init_launcher.add_field(1, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(locals[0].part_grad, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, locals[0].region_grad));
  init_launcher.add_field(2, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(locals[1].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, locals[1].region));
  init_launcher.add_field(3, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(locals[2].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, locals[2].region));
  init_launcher.add_field(4, FID_DATA);
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
  const Linear* linear = (Linear*) task->args;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  int input_channels = m->in_channels;
  int output_channels = m->out_channels;
  int batch_size = m->batch_size;
  const float *one_ptr = m->one_ptr;
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRO<float, 1> acc_kernel(regions[2], FID_DATA);
  const AccessorRO<float, 1> acc_bias(regions[3], FID_DATA);
  Rect<2> rect_input, rect_output;
  Rect<1> rect_kernel, rect_bias;
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
  //assert(acc_replica.accessor.is_dense_arbitrary(rect_replica));
  assert(acc_kernel.accessor.is_dense_arbitrary(rect_kernel));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);
  //float *replica_ptr = acc_output.ptr(rect_replica.lo);
  const float *kernel_ptr = acc_kernel.ptr(rect_kernel.lo);
  const float *bias_ptr = acc_bias.ptr(rect_bias.lo);
  //float *pre_relu_ptr = (m->relu) ? m->pre_relu : output_ptr;

  //checkCUDA(cudaMemcpy(replica_ptr, input_ptr, rect_input.volume() * sizeof(float),
  //                     cudaMemcpyDeviceToDevice));
  cudaEvent_t t_start, t_end;
  if (linear->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
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
  if (m->activation != AC_MODE_NONE) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
                                      &alpha, m->outputTensor, output_ptr,
                                      &beta, m->outputTensor, output_ptr));
  }
  if (linear->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Linear forward time = %.2lfms\n", elapsed);
  }
}

void Linear::forward(const FFModel& ff)
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
  IndexLauncher launcher(LINEAR_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Linear)), argmap);
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);
  //launcher.add_region_requirement(
  //    RegionRequirement(locals[0].partition, 0/*projection id*/,
  //                      WRITE_DISCARD, EXCLUSIVE, locals[0].region));
  //launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(locals[1].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, locals[1].region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(locals[2].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, locals[2].region));
  launcher.add_field(3, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): replica_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): filter
  regions[5](O): filter_grad
  regions[6](O): bias_grad
*/
__host__
void Linear::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);
  float alpha = 1.0f, beta = 0.0f;
  const Linear* linear = (Linear*) task->args;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  int input_channels = m->in_channels;
  int output_channels = m->out_channels;
  int batch_size = m->batch_size;
  const float *one_ptr = m->one_ptr;
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_replica_grad(regions[1], FID_DATA);
  const AccessorRO<float, 2> acc_output(regions[2], FID_DATA);
  const AccessorRW<float, 2> acc_output_grad(regions[3], FID_DATA);
  const AccessorRO<float, 1> acc_kernel(regions[4], FID_DATA);
  const AccessorWO<float, 2> acc_kernel_grad(regions[5], FID_DATA);
  const AccessorWO<float, 2> acc_bias_grad(regions[6], FID_DATA);
  Rect<2> rect_input, rect_replica_grad, rect_output, rect_output_grad,
          rect_kernel_grad, rect_bias_grad;
  Rect<1> rect_kernel;
  rect_input =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_replica_grad =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_output =
    runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_output_grad =
    runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  rect_kernel =
    runtime->get_index_space_domain(ctx, task->regions[4].region.get_index_space());
  rect_kernel_grad =
    runtime->get_index_space_domain(ctx, task->regions[5].region.get_index_space());
  rect_bias_grad =
    runtime->get_index_space_domain(ctx, task->regions[6].region.get_index_space());
  // make sure the sizes match
  assert(rect_input.volume() == input_channels * batch_size);
  assert(rect_replica_grad.volume() == input_channels * batch_size);
  assert(rect_output.volume() == output_channels * batch_size);
  assert(rect_output_grad.volume() == output_channels * batch_size);
  assert(rect_kernel.volume() == input_channels * output_channels);
  assert(rect_kernel_grad.volume() == input_channels * output_channels);
  assert(rect_bias_grad.volume() == output_channels);
  // make sure all regions are dense
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_replica_grad.accessor.is_dense_arbitrary(rect_replica_grad));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_output_grad.accessor.is_dense_arbitrary(rect_output_grad));
  assert(acc_kernel.accessor.is_dense_arbitrary(rect_kernel));
  assert(acc_kernel_grad.accessor.is_dense_arbitrary(rect_kernel_grad));
  assert(acc_bias_grad.accessor.is_dense_arbitrary(rect_bias_grad));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *replica_grad_ptr = acc_replica_grad.ptr(rect_replica_grad.lo);
  const float *output_ptr = acc_output.ptr(rect_output.lo);
  float *output_grad_ptr = acc_output_grad.ptr(rect_output_grad.lo);
  const float *kernel_ptr = acc_kernel.ptr(rect_kernel.lo);
  float *kernel_grad_ptr = acc_kernel_grad.ptr(rect_kernel_grad.lo);
  float *bias_grad_ptr = acc_bias_grad.ptr(rect_bias_grad.lo);

  cudaEvent_t t_start, t_end;
  if (linear->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  if (m->activation == AC_MODE_RELU) {
    int n = rect_output.volume();
    reluBackward<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(output_grad_ptr, output_ptr, n);
  } else {
    // TODO: only support relu or none activation
    assert(m->activation == AC_MODE_NONE);
  }

  // Compute weight gradiant
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        input_channels, output_channels, batch_size,
                        &alpha, input_ptr, input_channels,
                        output_grad_ptr, output_channels,
                        &beta, kernel_grad_ptr, input_channels));
  // Compute bias gradiant
  checkCUDA(cublasSgemv(m->handle.blas, CUBLAS_OP_N,
                        output_channels, batch_size,
                        &alpha, output_grad_ptr, output_channels,
                        one_ptr, 1,
                        &beta, bias_grad_ptr, 1));
  // Compute data gradiant
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        input_channels, batch_size, output_channels,
                        &alpha, kernel_ptr, input_channels,
                        output_grad_ptr, output_channels,
                        &beta, replica_grad_ptr, input_channels));
  if (linear->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Linear backward time = %.2lfms\n", elapsed);
  }
}

/*
  regions[0](O): input_grad
  regions[1..fc_num_par_c]: subreplicas
*/
__host__
void Linear::backward2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
  float alpha = 1.0f;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  const AccessorWO<float, 2> acc_input(regions[0], FID_DATA);
  Rect<2> rect_input, rect_replica;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  float *input_ptr = acc_input.ptr(rect_input.lo);
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));

  for (int i = 1; i < task->regions.size(); i++) {
    const AccessorRO<float, 2> acc_replica(regions[i], FID_DATA);
    rect_replica = runtime->get_index_space_domain(ctx, task->regions[i].region.get_index_space());
    //printf("rect_replica.hi = %lld lo = %lld\n", rect_replica.hi[0], rect_replica.lo[0]);
    //printf("rect_replica.hi = %lld lo = %lld\n", rect_replica.hi[1], rect_replica.lo[1]);
    //printf("rect_input.hi = %lld lo = %lld\n", rect_input.hi[0], rect_input.lo[0]);
    //printf("rect_input.hi = %lld lo = %lld\n", rect_input.hi[1], rect_input.lo[1]);
    assert(rect_replica.volume() == rect_input.volume());
    assert(acc_replica.accessor.is_dense_arbitrary(rect_replica));
    const float *replica_ptr = acc_replica.ptr(rect_replica.lo);
    if (i == 1)
      checkCUDA(cublasScopy(m->handle.blas, rect_input.volume(),
                            replica_ptr, 1, input_ptr, 1));
    else
      checkCUDA(cublasSaxpy(m->handle.blas, rect_input.volume(),
                            &alpha, replica_ptr, 1, input_ptr, 1));
  }
}

void Linear::backward(const FFModel& ff)
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
  {
    IndexLauncher launcher(LINEAR_BWD_TASK_ID, task_is,
                           TaskArgument(this, sizeof(Linear)), argmap);
    // regions[0](I): input
    launcher.add_region_requirement(
        RegionRequirement(input_lps[0], 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, inputs[0].region));
    launcher.add_field(0, FID_DATA);
    // regions[1](O): replica_grad (we only need grad tensors)
    launcher.add_region_requirement(
        RegionRequirement(locals[0].part_grad, 0/*projection id*/,
                          WRITE_DISCARD, EXCLUSIVE, locals[0].region_grad));
    launcher.add_field(1, FID_DATA);
    // regions[2](I): output
    launcher.add_region_requirement(
        RegionRequirement(output.part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, output.region));
    launcher.add_field(2, FID_DATA);
    // regions[3](I/O): output_grad
    launcher.add_region_requirement(
        RegionRequirement(output.part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, output.region_grad));
    launcher.add_field(3, FID_DATA);
    // regions[4](I): filter
    launcher.add_region_requirement(
        RegionRequirement(locals[1].part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, locals[1].region));
    launcher.add_field(4, FID_DATA);
    // regions[5](O): filter_grad
    launcher.add_region_requirement(
        RegionRequirement(locals[1].part_grad, 0/*projection id*/,
                          WRITE_DISCARD, EXCLUSIVE, locals[1].region_grad));
    launcher.add_field(5, FID_DATA);
    // regions[6](O): bias_grad
    launcher.add_region_requirement(
        RegionRequirement(locals[2].part_grad, 0/*projection id*/,
                          WRITE_DISCARD, EXCLUSIVE, locals[2].region_grad));
    launcher.add_field(6, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  {
    // We aggregate parameters from replica tensor to input tensor
    IndexLauncher launcher2(LINEAR_BWD2_TASK_ID, task_is,
                            TaskArgument(this, sizeof(Linear)), argmap);
    launcher2.add_region_requirement(
        RegionRequirement(inputs[0].part_grad, 0/*projection id*/,
                          WRITE_DISCARD, EXCLUSIVE, inputs[0].region_grad));
    launcher2.add_field(0, FID_DATA);
    for (int i = 0; i < fc_num_par_c; i++) {
      launcher2.add_region_requirement(
          RegionRequirement(replica_sub_lps[i], 0/*partition id*/,
                            READ_ONLY, EXCLUSIVE, locals[0].region_grad));
      launcher2.add_field(i + 1, FID_DATA);
    }
    runtime->execute_index_space(ctx, launcher2);
  }
}

#ifdef DEADCODE
/*
  regions[0](I/O): filter
  regions[1](I): filter_grad
  regions[2](I/O): bias
  regions[3](I): bias_grad
*/
__host__
void Linear::update_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const Linear* linear = (Linear*) task->args;
  const AccessorRW<float, 1> acc_filter(regions[0], FID_DATA);
  const AccessorRO<float, 2> acc_filter_grad(regions[1], FID_DATA);
  const AccessorRW<float, 1> acc_bias(regions[2], FID_DATA);
  const AccessorRO<float, 2> acc_bias_grad(regions[3], FID_DATA);
  Rect<1> rect_filter, rect_bias;
  Rect<2> rect_filter_grad, rect_bias_grad;
  rect_filter =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_filter_grad =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_bias =
    runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_bias_grad =
    runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  size_t filter_size = rect_filter.volume();
  size_t bias_size = rect_bias.volume();
  assert(filter_size == linear->in_channels * linear->out_channels);
  assert(bias_size == linear->out_channels);
  printf("filter_size(%d) linear->num_replica(%d) rect_filter_grad(%d)\n", filter_size, linear->num_replica, rect_filter_grad.volume());
  assert(filter_size * linear->num_replica == rect_filter_grad.volume());
  assert(bias_size * linear->num_replica == rect_bias_grad.volume());
  assert(acc_filter.accessor.is_dense_arbitrary(rect_filter));
  assert(acc_filter_grad.accessor.is_dense_arbitrary(rect_filter_grad));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  assert(acc_bias_grad.accessor.is_dense_arbitrary(rect_bias_grad));
  float *filter_ptr = acc_filter.ptr(rect_filter.lo);
  const float *filter_grad_ptr = acc_filter_grad.ptr(rect_filter_grad.lo);
  float *bias_ptr = acc_bias.ptr(rect_bias.lo);
  const float *bias_grad_ptr = acc_bias_grad.ptr(rect_bias_grad.lo);
  updateGAS(filter_ptr, filter_grad_ptr, filter_size,
            linear->num_replica, linear->learning_rate);
  updateGAS(bias_ptr, bias_grad_ptr, bias_size,
            linear->num_replica, linear->learning_rate);
}

__host__
void Linear::update(const FFModel& ff)
{
  // Synchronize the learning rate
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  learning_rate = ff.config.learningRate;
  assert(num_replica > 0);
  // Only aggregate parameters if more than one replica
  if (num_replica > 1) {
    TaskLauncher launcher(LINEAR_UPD_TASK_ID, TaskArgument(this, sizeof(Linear)));
    launcher.add_region_requirement(
      RegionRequirement(locals[1].region, READ_WRITE, EXCLUSIVE, locals[1].region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(locals[1].region_grad, READ_ONLY, EXCLUSIVE, locals[1].region_grad));
    launcher.add_field(1, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(locals[2].region, READ_WRITE, EXCLUSIVE, locals[2].region));
    launcher.add_field(2, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(locals[2].region_grad, READ_ONLY, EXCLUSIVE, locals[2].region_grad));
    launcher.add_field(3, FID_DATA);
    runtime->execute_task(ctx, launcher);
  }
}
#endif
