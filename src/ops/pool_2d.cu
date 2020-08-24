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

Tensor FFModel::pool2d(const Tensor& input,
                       int kernelH, int kernelW,
                       int strideH, int strideW,
                       int paddingH, int paddingW,
                       PoolType type, ActiMode activation)
{
  assert(input.numDim == 4); /*NCHW*/
  Pool2D *pool = new Pool2D(*this, input,kernelH, kernelW,
                            strideH, strideW, paddingH, paddingW,
                            type, activation);
  layers.push_back(pool);
  return pool->outputs[0];
}

Pool2D* FFModel::pool2d(int kernelH, int kernelW,
                        int strideH, int strideW,
                        int paddingH, int paddingW,
                        PoolType type, ActiMode activation)
{
  Pool2D *pool = new Pool2D(*this, kernelH, kernelW,
                            strideH, strideW, paddingH, paddingW,
                            type, activation);
  layers.push_back(pool);
  return pool;
}

Pool2D::Pool2D(FFModel& model,
               const Tensor& _input,
               int _kernel_h, int _kernel_w,
               int _stride_h, int _stride_w,
               int _padding_h, int _padding_w,
               PoolType _type, ActiMode _activation)
: Op(model, OP_POOL2D, "Pool2D_"+std::to_string(_kernel_h)+std::to_string(_kernel_w), _input),
  kernel_h(_kernel_h), kernel_w(_kernel_w),
  stride_h(_stride_h), stride_w(_stride_w),
  padding_h(_padding_h), padding_w(_padding_w),
  pool_type(_type), activation(_activation),
  profiling(model.config.profiling)
{
  int input_w = inputs[0].adim[0];
  int input_h = inputs[0].adim[1];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_c = inputs[0].adim[2];
  int output_n = inputs[0].adim[3];
  outputs[0].numDim = 4;
  outputs[0].adim[0] = output_w;
  outputs[0].adim[1] = output_h;
  outputs[0].adim[2] = output_c;
  outputs[0].adim[3] = output_n;
}

Pool2D::Pool2D(FFModel& model,
               int _kernel_h, int _kernel_w,
               int _stride_h, int _stride_w,
               int _padding_h, int _padding_w,
               PoolType _type, ActiMode _activation)
: Op(model, OP_POOL2D, "Pool2D_"+std::to_string(_kernel_h)+std::to_string(_kernel_w), 1),
  kernel_h(_kernel_h), kernel_w(_kernel_w),
  stride_h(_stride_h), stride_w(_stride_w),
  padding_h(_padding_h), padding_w(_padding_w),
  pool_type(_type), activation(_activation),
  profiling(model.config.profiling)
{
}

Tensor Pool2D::init_inout(FFModel& model, const Tensor& _input)
{
  inputs[0] = _input;
  create_output_and_partition(model);
  return outputs[0];
}


void Pool2D::create_weights(FFModel& model)
{
  // Do nothing since we don't have any weight
}

/*
void Pool2D::add_to_model(FFModel& model)
{
  model.layers.push_back(this);
}
*/

void Pool2D::create_output_and_partition(FFModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  std::string pcname = name;
  task_is = IndexSpaceT<4>(model.get_or_create_task_is(4, pcname));
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, task_is);

  int input_w = inputs[0].adim[0];
  int input_h = inputs[0].adim[1];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_c = inputs[0].adim[2];
  int output_n = inputs[0].adim[3];
  {
    const int dims[4] = {output_n, output_c, output_h, output_w};
    outputs[0] = model.create_tensor<4>(dims, (IndexSpaceT<4>)task_is, DT_FLOAT);
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }
  //int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  //int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  //int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;
  Rect<4> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
  //TODO: currently do not support splitting over the channel dimension
  assert(num_par_c == 1);
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0].part;
    input_grad_lps[0] = inputs[0].part_grad;
  } else {
    model.create_disjoint_partition(
        inputs[0], (IndexSpaceT<4>)task_is, input_lps[0], input_grad_lps[0]);
  }
}

/*
  regions[0]: input
  regions[1]: output
*/
OpMeta* Pool2D::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Pool2D* pool = (Pool2D*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  Pool2DMeta* m = new Pool2DMeta(handle);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);

  int input_w = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int input_h = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
  int input_c = acc_input.rect.hi[2] - acc_input.rect.lo[2] + 1;
  int input_n = acc_input.rect.hi[3] - acc_input.rect.lo[3] + 1;
  int output_w = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int output_h = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
  int output_c = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
  int output_n = acc_output.rect.hi[3] - acc_output.rect.lo[3] + 1;

  printf("init pool (input): n(%d) c(%d) h(%d) w(%d)\n",
         input_n, input_c, input_h, input_w);
  printf("init pool (output): n(%d) c(%d) h(%d) w(%d)\n",
         output_n, output_c, output_h, output_w);
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));
  int pad_h = ((output_h - 1) * pool->stride_h + pool->kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * pool->stride_w + pool->kernel_w - input_w + 1) / 2;
  if (pad_h != pool->padding_h)
    printf("Warning: changing pool_padding_h to satisfy output_h size\n");
  if (pad_w != pool->padding_w)
    printf("Warning: changing pool_padding_w to satisfy output_w size\n");
  
  cudnnPoolingMode_t mode;
  if (pool->pool_type == POOL_MAX)
    mode = CUDNN_POOLING_MAX;
  else {
    assert(pool->pool_type == POOL_AVG);
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  checkCUDNN(cudnnSetPooling2dDescriptor(m->poolDesc,
                                         mode,
                                         CUDNN_PROPAGATE_NAN,
                                         pool->kernel_h,
                                         pool->kernel_w,
                                         pad_h,//pool->padding_h,
                                         pad_w,//pool->padding_w,
                                         pool->stride_h,
                                         pool->stride_w));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(m->poolDesc,
                                               m->inputTensor,
                                               &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  return m;
}

void Pool2D::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<4> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher init_launcher(POOL2D_INIT_TASK_ID, task_is,
                              TaskArgument(this, sizeof(Pool2D)), argmap,
                              Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                              FFConfig::get_hash_id(std::string(name)));
  init_launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, outputs[0].region));
  init_launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/

void Pool2D::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  float alpha = 1.0f, beta = 0.0f;
  const Pool2DMeta* m = *((Pool2DMeta**) task->local_args);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  checkCUDNN(cudnnPoolingForward(m->handle.dnn, m->poolDesc,
                                 &alpha, m->inputTensor, acc_input.ptr,
                                 &beta, m->outputTensor, acc_output.ptr));
}

void Pool2D::forward(const FFModel& ff)
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
  IndexLauncher launcher(POOL2D_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Pool2D)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, outputs[0].region));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](I/O): input_grad
  regions[2](I): output
  regions[3](I): output_grad
*/
void Pool2D::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f;
  const Pool2D* pool = (Pool2D*) task->args;
  const Pool2DMeta* m = *((Pool2DMeta**) task->local_args);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_input_grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, 4> acc_output(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 4> acc_output_grad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);

  cudaEvent_t t_start, t_end;
  if (pool->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  checkCUDNN(cudnnPoolingBackward(m->handle.dnn, m->poolDesc,
                                  &alpha, m->outputTensor, acc_output.ptr,
                                  m->outputTensor, acc_output_grad.ptr,
                                  m->inputTensor, acc_input.ptr,
                                  &alpha, m->inputTensor, acc_input_grad.ptr));
  if (pool->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Pool2D backward time = %.2fms\n", elapsed);
  }
}

void Pool2D::backward(const FFModel& ff)
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
  IndexLauncher launcher(POOL2D_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Pool2D)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0](I): input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input_grad
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): output
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I): output_grad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(3, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

Pool2DMeta::Pool2DMeta(FFHandler handler)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
}

bool Pool2D::measure_compute_time(Simulator* sim,
                                  const ParallelConfig& pc,
                                  float& forward_time,
                                  float& backward_time)
{
  Tensor sub_output, sub_input;
  if(!outputs[0].get_output_sub_tensor(pc, sub_output, OP_CONV2D))
    return false;
  if(!inputs[0].get_input_sub_tensor(pc, sub_input, OP_CONV2D))
    return false;
  int input_w = sub_input.adim[0];
  int input_h = sub_input.adim[1];
  int input_c = sub_input.adim[2];
  int input_n = sub_input.adim[3];
  int output_w = sub_output.adim[0];
  int output_h = sub_output.adim[1];
  int output_c = sub_output.adim[2];
  int output_n = sub_output.adim[3];
  int pad_h = ((output_h - 1) * stride_h + kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * stride_w + kernel_w - input_w + 1) / 2;
  Pool2DMeta* m = sim->pool2d_meta;
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));
  cudnnPoolingMode_t mode;
  if (pool_type == POOL_MAX)
    mode = CUDNN_POOLING_MAX;
  else {
    assert(pool_type == POOL_AVG);
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  checkCUDNN(cudnnSetPooling2dDescriptor(m->poolDesc,
                                         mode,
                                         CUDNN_PROPAGATE_NAN,
                                         kernel_h,
                                         kernel_w,
                                         pad_h,//pool->padding_h,
                                         pad_w,//pool->padding_w,
                                         stride_h,
                                         stride_w));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(m->poolDesc,
                                               m->inputTensor,
                                               &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  // allocate tensors in simulator
  sim->free_all();
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  float* input_grad_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_grad_ptr != NULL);
  float *output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  float *output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_grad_ptr != NULL);

  float alpha = 1.0f, beta = 0.0f;
  // measure forward time
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < sim->warmup_times + sim->repeat_times; i++) {
    if (i == sim->warmup_times) {
      checkCUDA(cudaEventRecord(sim->start_event));
    }
    checkCUDNN(cudnnPoolingForward(m->handle.dnn, m->poolDesc,
                                   &alpha, m->inputTensor, input_ptr,
                                   &beta, m->outputTensor, output_ptr));

  }
  checkCUDA(cudaEventRecord(sim->end_event));
  checkCUDA(cudaEventSynchronize(sim->end_event));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, sim->start_event, sim->end_event);
  forward_time = milliseconds / sim->repeat_times;

  // measure backward time
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < sim->warmup_times + sim->repeat_times; i++) {
    if (i == sim->warmup_times) {
      checkCUDA(cudaEventRecord(sim->start_event));
    }
    checkCUDNN(cudnnPoolingBackward(m->handle.dnn, m->poolDesc,
                                    &alpha, m->outputTensor, output_ptr,
                                    m->outputTensor, output_grad_ptr,
                                    m->inputTensor, input_ptr,
                                    &alpha, m->inputTensor, input_grad_ptr));
  }
  checkCUDA(cudaEventRecord(sim->end_event));
  checkCUDA(cudaEventSynchronize(sim->end_event));
  cudaEventElapsedTime(&milliseconds, sim->start_event, sim->end_event);
  backward_time = milliseconds / sim->repeat_times;

  return false;
}

