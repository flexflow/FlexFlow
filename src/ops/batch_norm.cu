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

#include "model.h"
#include "cuda_helper.h"

using namespace Legion;

Tensor FFModel::batch_norm(const Tensor input,
                           bool relu,
                           const char* name)
{
  assert(input->num_dims == 4); //Only support 4D BN for now
  Initializer* scale_initializer = new ConstantInitializer(1.0f);
  Initializer* bias_initializer = new ConstantInitializer(0.0f);
#ifdef FF_USE_NCCL
  ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
  ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
  Tensor scale, bias;
  {
    const int dims[1] = {input->dims[2].size};
    scale = create_weight<1>(dims, DT_FLOAT, NULL/*owner_op*/,
        true/*create_grad*/, scale_initializer, comm_type);
  }
  {
    const int dims[1] = {input->dims[2].size};
    bias = create_weight<1>(dims, DT_FLOAT, NULL/*owner_op*/,
        true/*create_grad*/, bias_initializer, comm_type);
  }
  BatchNorm *bn = new BatchNorm(*this, input, scale, bias, relu, name);
  layers.push_back(bn);
  return bn->outputs[0];
}

/*
  locals[0] = scale
  locals[1] = bias
*/
BatchNorm::BatchNorm(FFModel& model,
                     const Tensor _input,
                     const Tensor _scale,
                     const Tensor _bias,
                     bool _relu,
                     const char* name)
: Op(model, OP_BATCHNORM, name, 1/*inputs*/, 2/*weights*/, _input, _scale, _bias),
  relu(_relu)
{
  assert(_input->num_dims == 4);
  numOutputs = 1;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < _input->num_dims; i++)
    dims[i] = _input->dims[_input->num_dims-1-i];
  outputs[0] = model.create_tensor(_input->num_dims, dims, DT_FLOAT, this);
  return;
}

#ifdef DEADCODE
void BatchNorm::create_weights(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<4>(model.get_or_create_task_is(4, pcname));
  // Create scale and bias
  Initializer* scale_initializer = new ConstantInitializer(1.0f);
  Initializer* bias_initializer = new ConstantInitializer(0.0f);
  const int dims[1] = {outputs[0].adim[2]};
  weights[0] = model.create_conv_weight<1>(this, dims, DT_FLOAT, scale_initializer);
  weights[1] = model.create_conv_weight<1>(this, dims, DT_FLOAT, bias_initializer);
}
#endif

#ifdef DEADCODE
void BatchNorm::create_input_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<4>(model.get_or_create_task_is(4, pcname));

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;
  // Currently assume data parallelism for batch norm
  assert(num_par_w == 1);
  assert(num_par_h == 1);
  assert(num_par_c == 1);
  return Op::create_input_partition(model);
#ifdef DEADCODE
  // Create output tensor
  int output_w = outputs[0].dims[0].size;
  int output_h = outputs[0].dims[1].size;
  int output_c = outputs[0].dims[2].size;
  int output_n = outputs[0].dims[3].size;
  {
    const int dims[4] = {output_n, output_c, output_h, output_w};
    outputs[0] = model.create_tensor<4>(dims, DT_FLOAT, this);
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }
  // Compute partition bound for input
  Rect<4> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0]->part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0]->part;
    input_grad_lps[0] = inputs[0]->part_grad;
  } else {
    model.create_disjoint_partition(
        inputs[0], (IndexSpaceT<4>)task_is, input_lps[0], input_grad_lps[0]);
  }
#endif
}
#endif

/*
  regions[0]: input
  regions[1]: output
  regions[2](I): scale
  regions[3](I): bias
*/
__host__
OpMeta* BatchNorm::init_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const BatchNorm* bm = (BatchNorm*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 1> acc_scale(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 1> acc_bias(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);

  int output_w = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int output_h = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
  int output_c = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
  int output_n = acc_output.rect.hi[3] - acc_output.rect.lo[3] + 1;

  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
      .only_kind(Memory::GPU_FB_MEM).best_affinity_to(task->target_proc).first();
  BatchNormMeta* m = new BatchNormMeta(handle, bm, gpu_mem,
      output_n, output_c, output_h, output_w);
  return m;
}

#ifdef DEADCODE
/*
  regions[0](O): scale, initilized to ones
  regions[1](O): bias, initilized to zeros
*/
__host__
void BatchNorm::init_para_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const BatchNorm* bm = (BatchNorm*) task->args;
  const AccessorWO<float, 1> acc_scale(regions[0], FID_DATA);
  const AccessorWO<float, 1> acc_bias(regions[1], FID_DATA);
  Rect<1> rect_scale, rect_bias;
  rect_scale = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_scale.accessor.is_dense_arbitrary(rect_scale));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  float *scale_ptr = acc_scale.ptr(rect_scale.lo);
  float *bias_ptr = acc_bias.ptr(rect_bias.lo);
  // init kernel and bias
#ifdef PARAMETER_ALL_ONES
  ones_kernel<<<GET_BLOCKS(rect_scale.volume()), CUDA_NUM_THREADS>>>(
      scale_ptr, rect_scale.volume());
  ones_kernel<<<GET_BLOCKS(rect_bias.volume()), CUDA_NUM_THREADS>>>(
      bias_ptr, rect_bias.volume());
#else
  //cudaStream_t stream;
  //checkCUDA(cudaStreamCreate(&stream));
  //curandGenerator_t genGPU;
  //curandCreateGenerator(&genGPU, CURAND_RNG_PSEUDO_DEFAULT);
  //curandSetStream(genGPU, stream);
  //curandSetPseudoRandomGeneratorSeed(genGPU, 1234ULL);
  //curandGenerateUniform(genGPU, scale_ptr, rect_scale.volume());
  assign_kernel<<<GET_BLOCKS(rect_scale.volume()), CUDA_NUM_THREADS>>>(
      scale_ptr, rect_scale.volume(), 1.0f);
  assign_kernel<<<GET_BLOCKS(rect_bias.volume()), CUDA_NUM_THREADS>>>(
      bias_ptr, rect_bias.volume(), 0.0f);
  //curandDestroyGenerator(genGPU);
#endif
}
#endif

__host__
void BatchNorm::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<4> rect = runtime->get_index_space_domain(ctx, task_is);
  ParallelConfig pc;
  std::string pcname = name;
  ff.config.find_parallel_config(4, pcname, pc);
  int idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[pc.device_ids[idx++]];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher launcher(BATCHNORM_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(BatchNorm)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[1]->region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[1]->region));
  launcher.add_field(3, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*static*/
void BatchNorm::forward_kernel(BatchNormMeta *m,
                               float const *input_ptr,
                               float *output_ptr,
                               float const *scale_ptr,
                               float const *bias_ptr)
{
  float alpha = 1.0f, beta = 0.0f;
  //coord_t numChannels = m->numChannels;
  checkCUDNN(cudnnBatchNormalizationForwardTraining(
             m->handle.dnn, m->mode, &alpha, &beta, m->inputTensor, input_ptr,
             m->outputTensor, output_ptr, m->biasTensor, scale_ptr, bias_ptr,
             1.0, m->runningMean, m->runningVar, CUDNN_BN_MIN_EPSILON,
             m->saveMean, m->saveVar));
}

/*
  regions[0](I): input
  regions[1](O): ouptut
  regions[2](I): scale
  regions[3](I): bias
*/
__host__
void BatchNorm::forward_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  //const BatchNorm* bm = (BatchNorm*) task->args;
  BatchNormMeta* m = *((BatchNormMeta**) task->local_args);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 1> acc_scale(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 1> acc_bias(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  forward_kernel(m, acc_input.ptr, acc_output.ptr, acc_scale.ptr, acc_bias.ptr);
  if (m->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("BatchNorm forward time (BF) = %.2fms\n", elapsed);
  }
}

__host__
void BatchNorm::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<4> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(BATCHNORM_FWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[1]->region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[1]->region));
  launcher.add_field(3, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

/*static*/
void BatchNorm::backward_kernel(BatchNormMeta *m,
                                float const *input_ptr,
                                float *output_grad_ptr,
                                float const *output_ptr,
                                float *input_grad_ptr,
                                float const *scale_ptr,
                                float *scale_grad_ptr,
                                float *bias_grad_ptr,
                                size_t numElements)
{
  float alpha = 1.0f;
  if (m->relu) {
    reluBackward<<<GET_BLOCKS(numElements), CUDA_NUM_THREADS>>>(output_grad_ptr, output_ptr, numElements);
  }
  checkCUDNN(cudnnBatchNormalizationBackward(
             m->handle.dnn, m->mode, &alpha, &alpha, &alpha, &alpha,
             m->inputTensor, input_ptr, m->outputTensor, output_grad_ptr,
             m->inputTensor, input_grad_ptr, m->biasTensor, scale_ptr,
             scale_grad_ptr, bias_grad_ptr, CUDNN_BN_MIN_EPSILON,
             m->saveMean, m->saveVar));
}

/*
  regions[0](I): input
  regions[1](I/O): input_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): scale
  regions[5](I/O): scale_grad
  regions[6](I/O): bias_grad
*/
__host__
void BatchNorm::backward_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);
  //float beta = 0.0f;
  //const BatchNorm* bm = (BatchNorm*) task->args;
  BatchNormMeta* m = *((BatchNormMeta**) task->local_args);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_input_grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, 4> acc_output(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_output_grad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, 1> acc_scale(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 1> acc_scale_grad(
      regions[5], task->regions[5], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorW<float, 1> acc_bias_grad(
      regions[6], task->regions[6], FID_DATA, ctx, runtime,
      true/*readOutput*/);

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  backward_kernel(m, acc_input.ptr, acc_output_grad.ptr, acc_output.ptr, acc_input_grad.ptr, acc_scale.ptr, acc_scale_grad.ptr, acc_bias_grad.ptr, acc_output.rect.volume());
  if (m->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("BatchNorm backward time = %.2fms\n", elapsed);
  }
}

__host__
void BatchNorm::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<4> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }

  IndexLauncher launcher(BATCHNORM_BWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0](I): input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input_grad (we only need grad tensors)
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): output
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I/O): output_grad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(3, FID_DATA);
  // regions[4](I): filter
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(4, FID_DATA);
  // regions[5](I/O): filter_grad
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, weights[0]->region_grad));
  launcher.add_field(5, FID_DATA);
  // regions[6](I/O): bias_grad
  launcher.add_region_requirement(
      RegionRequirement(weights[1]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, weights[1]->region_grad));
  launcher.add_field(6, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
}

BatchNormMeta::BatchNormMeta(FFHandler handler,
                             const BatchNorm* bn,
                             Memory gpu_mem,
                             int output_n,
                             int output_c,
                             int output_h,
                             int output_w)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  relu = bn->relu;
  profiling = bn->profiling;
  mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7000
  mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif
  fprintf(stderr, "output(%d,%d,%d,%d)\n",
    output_n, output_c, output_h, output_w);
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        output_n, output_c,
                                        output_h, output_w));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        output_n, output_c,
                                        output_h, output_w));
  checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        1, output_c, 1, 1));
  // allocate memory for runningMean, runningVar, saveMean, saveVar
  {
    size_t totalSize = sizeof(float) * output_c * 4;
    Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
        Realm::Point<1, coord_t>(totalSize-1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance::create_instance(reserveInst, gpu_mem, bounds,
        field_sizes, 0, Realm::ProfilingRequestSet()).wait();
    runningMean = (float*) reserveInst.pointer_untyped(0, sizeof(char));
    runningVar = (float*) runningMean + output_c;
    saveMean = (float*) runningVar + output_c;
    saveVar = (float*) saveMean + output_c;
    assign_kernel<<<GET_BLOCKS(output_c), CUDA_NUM_THREADS>>>(
      runningMean, output_c, 0.0f);
    assign_kernel<<<GET_BLOCKS(output_c), CUDA_NUM_THREADS>>>(
      runningVar, output_c, 0.0f);
  }
  if (relu) {
    checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
  }
}

BatchNormMeta::~BatchNormMeta(void)
{
  reserveInst.destroy();
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(biasTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  if (relu) {
    checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  }
}

bool BatchNorm::measure_operator_cost(Simulator* sim,
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

  int output_w = sub_output.dims[0].size;
  int output_h = sub_output.dims[1].size;
  int output_c = sub_output.dims[2].size;
  int output_n = sub_output.dims[3].size;
  BatchNormMeta *m = new BatchNormMeta(sim->handler, this, sim->memory,
      output_n, output_c, output_h, output_w);

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);
  float *bias_ptr = (float *)sim->allocate(output_c, DT_FLOAT);
  assert (bias_ptr != NULL);
  float *scale_ptr = (float *)sim->allocate(output_c, DT_FLOAT);
  assert (scale_ptr != NULL);
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, scale_ptr, bias_ptr);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert (input_grad_ptr != NULL);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    float *scale_grad_ptr = (float *)sim->allocate(output_c, DT_FLOAT);
    assert (scale_grad_ptr != NULL);
    float *bias_grad_ptr = (float *)sim->allocate(output_c, DT_FLOAT);
    assert (bias_grad_ptr != NULL);

    backward = [&] {
      backward_kernel(m, input_ptr, output_grad_ptr, output_ptr, input_grad_ptr,
          scale_ptr, scale_grad_ptr, bias_grad_ptr, sub_output.get_volume());
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure BatchNorm] name(%s) size(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
        name, sub_input.get_volume(),
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure BatchNorm] name(%s) size(%zu) forward_time(%.4lf)\n",
        name, sub_input.get_volume(),
        cost_metrics.forward_time);
  }
  // Free batchnormmeta
  delete m;
  return true;
}
