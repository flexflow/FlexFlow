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

Tensor FFModel::pool2d(std::string name, Tensor input,
                         int kernelH, int kernelW,
                         int strideH, int strideW,
                         int paddingH, int paddingW,
                         PoolType type, bool relu)
{
  assert(input.numDim == 4); /*NCHW*/
  ParallelConfig pc;
  assert(config.find_parallel_config(name, pc));
  IndexSpaceT<4> task_is = IndexSpaceT<4>(get_or_create_task_is(pc));
  Pool2D *pool = new Pool2D(name, config, input, task_is, kernelH, kernelW,
                            strideH, strideW, paddingH, paddingW,
                            type, relu);
  layers.push_back(pool);
  return pool->output;
}

Pool2D::Pool2D(std::string _name, FFConfig _config,
               Tensor _input, IndexSpaceT<4> _task_is,
               int _kernel_h, int _kernel_w,
               int _stride_h, int _stride_w,
               int _padding_h, int _padding_w,
               PoolType _type, bool _relu)
: Op(_name, _input), task_is(_task_is),
  kernel_h(_kernel_h), kernel_w(_kernel_w),
  stride_h(_stride_h), stride_w(_stride_w),
  padding_h(_padding_h), padding_w(_padding_w),
  pool_type(_type), relu(_relu), profiling(_config.profiling)
{
  Context ctx = _config.lg_ctx;
  Runtime* runtime = _config.lg_hlr;

  int input_w = _input.adim[0];
  int input_h = _input.adim[1];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_c = _input.adim[2];
  int output_n = _input.adim[3];
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;

  FieldSpace fs = _config.field_space;

  Rect<4> output_rect(Point<4>(0, 0, 0, 0),
      Point<4>(output_w-1, output_h-1, output_c-1, output_n-1));
  IndexSpaceT<4> output_is = runtime->create_index_space(ctx, output_rect);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  LogicalRegion output_grad_lr = runtime->create_logical_region(ctx, output_is, fs);
  int extent_w = (output_w + num_par_w - 1) / num_par_w;
  int extent_h = (output_h + num_par_h - 1) / num_par_h;
  int extent_c = output_c / num_par_c;
  int extent_n = output_n / num_par_n;
  assert(output_c % num_par_c == 0);
  assert(output_n % num_par_n == 0);
  Rect<4> extent(Point<4>(0, 0, 0, 0),
      Point<4>(extent_w-1, extent_h-1, extent_c-1, extent_n-1));
  Transform<4, 4> transform;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      transform[i][j] = 0;
  transform[0][0] = extent_w; 
  transform[1][1] = extent_h;
  transform[2][2] = extent_c;
  transform[3][3] = extent_n;
  IndexPartition output_ip =
    runtime->create_partition_by_restriction(ctx, output_is, task_is, transform, extent);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  LogicalPartition output_grad_lp =
    runtime->get_logical_partition(ctx, output_grad_lr, output_ip);

  output.numDim = 4;
  output.adim[0] = output_w;
  output.adim[1] = output_h;
  output.adim[2] = output_c;
  output.adim[3] = output_n;
  output.pdim[0] = extent_w;
  output.pdim[1] = extent_h;
  output.pdim[2] = extent_c;
  output.pdim[3] = extent_n;
  output.region = output_lr;
  output.part = output_lp;
  output.region_grad = output_grad_lr;
  output.part_grad = output_grad_lp;
  printf("Create pool2d layer: output(n=%d c=%d h=%d w=%d)\n",
         output.adim[3], output.adim[2], output.adim[1], output.adim[0]);

  // Compute partition bound for input
  Rect<4> input_part_rect =
    runtime->get_index_partition_color_space(ctx, inputs[0].part.get_index_partition());
  if (input_part_rect == part_rect) {
    input_lps[0] = _input.part;
  } else {
    printf("WARNING: input has a different partition!!!\n");
    IndexSpaceT<3> input_is = IndexSpaceT<3>(inputs[0].region.get_index_space());
    //extent_w = stride_w * (output.pdim[0]-1) + kernel_w - 2 * padding_w;
    //extent_h = stride_h * (output.pdim[1]-1) + kernel_h - 2 * padding_h;
    //extent_nc = inputs[0].adim[2] * inputs[0].adim[3] / config.num_par_n;
    extent_w = (inputs[0].adim[0] + num_par_w - 1) / num_par_w;
    extent_h = (inputs[0].adim[1] + num_par_h - 1) / num_par_h;
    extent_c = inputs[0].adim[2] / num_par_c;
    extent_n = inputs[0].adim[3] / num_par_n;
    assert(inputs[0].adim[2] % num_par_c == 0);
    assert(inputs[0].adim[3] % num_par_n == 0);
    Rect<4> extent_i(Point<4>(0, 0, 0, 0),
        Point<4>(extent_w-1, extent_h-1, extent_c-1, extent_n-1));
    //transform[0][0] = stride_w * output.pdim[0];
    //transform[1][1] = stride_h * output.pdim[1];
    //transform[2][2] = extent_nc;
    transform[0][0] = extent_w;
    transform[1][1] = extent_h;
    transform[2][2] = extent_c;
    transform[3][3] = extent_n;

    IndexPartition input_ip =
      runtime->create_partition_by_restriction(ctx, input_is, task_is, transform, extent_i);
    assert(runtime->is_index_partition_disjoint(ctx, input_ip));
    assert(runtime->is_index_partition_complete(ctx, input_ip));
    input_lps[0] = runtime->get_logical_partition(ctx, inputs[0].region, input_ip);
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
  assert(regions.size() == 2);
  const Pool2D* pool = (Pool2D*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  Pool2DMeta* m = new Pool2DMeta(handle);
  Rect<4> rect_input, rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  checkCUDNN(cudnnCreatePoolingDescriptor(&m->poolDesc));

  int input_w = rect_input.hi[0] - rect_input.lo[0] + 1;
  int input_h = rect_input.hi[1] - rect_input.lo[1] + 1;
  int output_w = rect_output.hi[0] - rect_output.lo[0] + 1;
  int output_h = rect_output.hi[1] - rect_output.lo[1] + 1;
  printf("init pool (input): n(%d) c(%d) h(%d) w(%d)\n", pool->inputs[0].pdim[3],
        pool->inputs[0].pdim[2], input_h, input_w);
  printf("init pool (output): n(%d) c(%d) h(%d) w(%d)\n", pool->output.pdim[3],
        pool->output.pdim[2], output_h, output_w);
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        pool->inputs[0].pdim[3],
                                        pool->inputs[0].pdim[2],
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
  assert(n == pool->output.pdim[3]);
  assert(c == pool->output.pdim[2]);
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
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
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
  const AccessorRO<float, 4> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 4> acc_output(regions[1], FID_DATA);
  Rect<4> rect_input, rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  checkCUDNN(cudnnPoolingForward(m->handle.dnn, m->poolDesc,
                                 &alpha, m->inputTensor, input_ptr,
                                 &beta, m->outputTensor, output_ptr));
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
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): input_grad
  regions[2](I): output
  regions[3](I): output_grad
*/
void Pool2D::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f, beta = 0.0f;
  const Pool2D* pool = (Pool2D*) task->args;
  const Pool2DMeta* m = *((Pool2DMeta**) task->local_args);
  const AccessorRO<float, 4> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 4> acc_input_grad(regions[1], FID_DATA);
  const AccessorRO<float, 4> acc_output(regions[2], FID_DATA);
  const AccessorRO<float, 4> acc_output_grad(regions[3], FID_DATA);
  Rect<4> rect_input, rect_input_grad, rect_output, rect_output_grad;
  rect_input =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_input_grad =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_output =
    runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_output_grad =
    runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_input_grad.accessor.is_dense_arbitrary(rect_input_grad));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_output_grad.accessor.is_dense_arbitrary(rect_output_grad));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *input_grad_ptr = acc_input_grad.ptr(rect_input_grad.lo);
  const float *output_ptr = acc_output.ptr(rect_output.lo);
  const float *output_grad_ptr = acc_output_grad.ptr(rect_output_grad.lo);

  cudaEvent_t t_start, t_end;
  if (pool->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  checkCUDNN(cudnnPoolingBackward(m->handle.dnn, m->poolDesc,
                                  &alpha, m->outputTensor, output_ptr,
                                  m->outputTensor, output_grad_ptr,
                                  m->inputTensor, input_ptr,
                                  &beta, m->inputTensor, input_grad_ptr));
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
  // regions[1](O): input_grad
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part_grad, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): output
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I): output_grad
  launcher.add_region_requirement(
      RegionRequirement(output.part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, output.region_grad));
  launcher.add_field(3, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

void Pool2D::update(const FFModel& ff)
{
}
