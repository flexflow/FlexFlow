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
#include <unistd.h>
#define PARAMETER_ALL_ONES

Tensor CnnModel::add_conv_layer(Tensor input, int out_channels,
                                int kernel_x, int kernel_y,
                                int stride_x, int stride_y,
                                int padding_x, int padding_y, bool relu)
{
  assert(input.numDim == 4); /*NCHW*/
  int in_channels = input.adim[2];
  Conv2D *conv = new Conv2D(config, input, part_is,
                            in_channels, out_channels, kernel_x, kernel_y,
                            stride_x, stride_y, padding_x, padding_y, relu);
  layers.push_back(conv);
  return conv->output;
}

/*
locals[0] = kernel
locals[1] = bias
*/
Conv2D::Conv2D(CnnConfig config, Tensor input, IndexSpaceT<3> part_is,
               int _in_channels, int _out_channels,
               int _kernel_h, int _kernel_w,
               int _stride_h, int _stride_w,
               int _padding_h, int _padding_w, bool _relu)
: Op(input), in_channels(_in_channels), out_channels(_out_channels),
  kernel_h(_kernel_h), kernel_w(_kernel_w), stride_h(_stride_h),
  stride_w(_stride_w), padding_h(_padding_h), padding_w(_padding_w), relu(_relu)
{
  Context ctx = config.lg_ctx;
  HighLevelRuntime* runtime = config.lg_hlr;
  // Create output tensor
  int input_w = input.adim[0];
  int input_h = input.adim[1];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_nc = input.adim[3] * out_channels;
  FieldSpace fs;
  fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }

  Rect<3, coord_t> output_rect(Point<3>(0, 0, 0),
                      Point<3>(output_w-1, output_h-1, output_nc-1));
  IndexSpaceT<3> output_is = runtime->create_index_space(ctx, output_rect);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  Transform<3, 3, coord_t> transform;
  int extent_w = (output_w + config.num_par_w - 1) / config.num_par_w;
  int extent_h = (output_h + config.num_par_h - 1) / config.num_par_h;
  int extent_nc = output_nc / config.num_par_n;
  assert(output_nc % config.num_par_n == 0);
  Rect<3, coord_t> extent(Point<3>(0, 0, 0), Point<3>(extent_w-1, extent_h-1, extent_nc-1));
  transform[0][0] = extent_w; transform[0][1] = 0; transform[0][2] = 0;
  transform[1][0] = 0; transform[1][1] = extent_h; transform[1][2] = 0;
  transform[2][0] = 0; transform[2][1] = 0; transform[2][2] = extent_nc;
  IndexPartition output_ip =
    runtime->create_partition_by_restriction(ctx, output_is, part_is, transform, extent);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);

  int kernel_nc = config.num_workers * in_channels * out_channels;
  Rect<1, coord_t> kernel_rect(0, kernel_w * kernel_h * kernel_nc - 1);
  IndexSpaceT<1> kernel_is = runtime->create_index_space(ctx, kernel_rect);
  LogicalRegion kernel_lr = runtime->create_logical_region(ctx, kernel_is, fs);
  IndexPartition kernel_ip = runtime->create_equal_partition(ctx, kernel_is, part_is);
  LogicalPartition kernel_lp = runtime->get_logical_partition(ctx, kernel_lr, kernel_ip);
  TensorWithGrad kernel_tensor;
  kernel_tensor.region = kernel_lr;
  kernel_tensor.partition = kernel_lp;
  locals[0] = kernel_tensor;

  int bias_nc = config.num_workers * out_channels;
  Rect<1, coord_t> bias_rect(0, bias_nc - 1);
  IndexSpaceT<1> bias_is = runtime->create_index_space(ctx, bias_rect);
  LogicalRegion bias_lr = runtime->create_logical_region(ctx, bias_is, fs);
  IndexPartition bias_ip = runtime->create_equal_partition(ctx, bias_is, part_is);
  LogicalPartition bias_lp = runtime->get_logical_partition(ctx, bias_lr, bias_ip);
  TensorWithGrad bias_tensor;
  bias_tensor.region = bias_lr;
  bias_tensor.partition = bias_lp;
  locals[1] = bias_tensor;

  output.numDim = 4;
  output.adim[0] = output_w;
  output.adim[1] = output_h;
  output.adim[2] = out_channels;
  output.adim[3] = input.adim[3];
  output.pdim[0] = extent_w;
  output.pdim[1] = extent_h;
  output.pdim[2] = out_channels;
  output.pdim[3] = input.adim[3];
  output.region = output_lr;
  output.partition = output_lp;
  printf("Create conv layer: output(n=%d c=%d h=%d w=%d)\n",
         output.adim[3], output.adim[2], output.adim[1], output.adim[0]);

  // Compute partition bound for input
  IndexSpaceT<3> input_is = IndexSpaceT<3>(inputs[0].region.get_index_space());
  extent_w = stride_w * (output.pdim[0]-1) + kernel_w - 2 * padding_w;
  extent_h = stride_h * (output.pdim[1]-1) + kernel_h - 2 * padding_h;
  extent_nc = inputs[0].adim[2] * inputs[0].adim[3] / config.num_par_n;
  assert(inputs[0].adim[2] * inputs[0].adim[3] % config.num_par_n == 0);
  Rect<3, coord_t> extent_i(Point<3>(0, 0, 0), Point<3>(extent_w-1, extent_h-1, extent_nc-1));
  transform[0][0] = stride_w * output.pdim[0];
  transform[1][1] = stride_h * output.pdim[1];
  transform[2][2] = extent_nc;
  IndexPartition input_ip =
    runtime->create_partition_by_restriction(ctx, input_is, part_is, transform, extent_i);
  input_lps[0] = runtime->get_logical_partition(ctx, inputs[0].region, input_ip);
}

cudnnConvolutionFwdAlgo_t
selectConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                  const cudnnTensorDescriptor_t xDesc, const void* x,
                                  const cudnnFilterDescriptor_t wDesc, const void* w,
                                  const cudnnConvolutionDescriptor_t convDesc,
                                  void* workSpace, size_t workSpaceSize,
                                  const cudnnTensorDescriptor_t yDesc, void* y);
/*
  regions[0]: input
  regions[1]: output
  regions[2]: filter
  regions[3]: bias
*/
__host__
OpMeta* Conv2D::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  const int BLKSIZE = 512;
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const Conv2D* conv = (Conv2D*) task->args;
  CnnHandle handle = *((const CnnHandle*) task->local_args);
  const AccessorRO<float, 3> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 3> acc_output(regions[1], FID_DATA);
  const AccessorWO<float, 1> acc_filter(regions[2], FID_DATA);
  const AccessorWO<float, 1> acc_bias(regions[3], FID_DATA);
  Rect<1> rect_filter, rect_bias;
  Rect<3> rect_input, rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_filter = runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_filter.accessor.is_dense_arbitrary(rect_filter));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);
  float *filter_ptr = acc_filter.ptr(rect_filter.lo);
  float *bias_ptr = acc_bias.ptr(rect_bias.lo);

#ifdef PARAMETER_ALL_ONES
  coord_t filter_elements = conv->inputs[0].adim[2] * conv->output.adim[2] * conv->kernel_h * conv->kernel_w;
  int num_blocks = (filter_elements + BLKSIZE - 1) / BLKSIZE;
  ones_kernel<<<num_blocks, BLKSIZE>>>(filter_ptr, filter_elements);
  num_blocks = (conv->output.pdim[2] + BLKSIZE - 1) / BLKSIZE;
  ones_kernel<<<num_blocks, BLKSIZE>>>(bias_ptr, conv->output.pdim[2]);
#else
  curandGenerator_t genGPU;
  curandCreateGenerator(&genGPU, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(genGPU, 1234ULL);
  coord_t filter_elements = conv->inputs[0].adim[2] * conv->output.adim[2] 
                          * conv->kernel_h * conv->kernel_w;
  float factor = 1.0f / sqrt(filter_elements / conv->output.adim[2]);
  printf("factor = %.4f elements = %d\n", factor, filter_elements / conv->output.adim[2]);
  assert(filter_elements == (coord_t) rect_filter.volume());
  curandGenerateUniform(genGPU, filter_ptr, filter_elements);
  int num_blocks = (filter_elements + BLKSIZE - 1) / BLKSIZE;
  scale_kernel<<<num_blocks, BLKSIZE>>>(filter_ptr, filter_elements, -factor, factor);
  curandGenerateUniform(genGPU, bias_ptr, conv->output.pdim[2]);
  num_blocks = (conv->output.pdim[2] + BLKSIZE - 1) / BLKSIZE;
  scale_kernel<<<num_blocks, BLKSIZE>>>(bias_ptr, conv->output.pdim[2], -factor, factor);
  curandDestroyGenerator(genGPU);
#endif
  Conv2DMeta* m = new Conv2DMeta(handle);
  m->relu = conv->relu;
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&m->biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&m->filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&m->convDesc));

  int input_w = rect_input.hi[0] - rect_input.lo[0] + 1;
  int input_h = rect_input.hi[1] - rect_input.lo[1] + 1;
  int output_w = rect_output.hi[0] - rect_output.lo[0] + 1;
  int output_h = rect_output.hi[1] - rect_output.lo[1] + 1;
  printf("init conv (input): n(%d) c(%d) h(%d) w(%d)\n", conv->inputs[0].pdim[3],
         conv->inputs[0].pdim[2], input_h, input_w);
  printf("init conv (output): n(%d) c_out(%d) h(%d) w(%d)\n", conv->output.pdim[3],
         conv->output.pdim[2], output_h, output_w);
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        conv->inputs[0].pdim[3],
                                        conv->inputs[0].pdim[2],
                                        input_h,
                                        input_w));
  
  checkCUDNN(cudnnSetTensor4dDescriptor(m->biasTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        1,
                                        conv->output.pdim[2],
                                        1,
                                        1));

  printf("filterDim: kernel(%d %d) c_out(%d)\n", conv->kernel_h, conv->kernel_w, conv->output.pdim[2]);
  checkCUDNN(cudnnSetFilter4dDescriptor(m->filterDesc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        conv->output.pdim[2],
                                        conv->inputs[0].pdim[2],
                                        conv->kernel_h,
                                        conv->kernel_w));

  printf("convDim: padding(%d %d) stride(%d %d)\n", conv->padding_h, conv->padding_w, conv->stride_h, conv->stride_w);
  checkCUDNN(cudnnSetConvolution2dDescriptor(m->convDesc,
                                             conv->padding_h,
                                             conv->padding_w,
                                             conv->stride_h,
                                             conv->stride_w,
                                             1/*upscale_x*/,
                                             1/*upscale_y*/,
                                             CUDNN_CROSS_CORRELATION));

  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(m->convDesc,
                                                   m->inputTensor,
                                                   m->filterDesc,
                                                   &n, &c, &h, &w));
  assert(n == conv->output.pdim[3]);
  assert(c == conv->output.pdim[2]);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  // select forward algorithm
  m->fwdAlgo = selectConvolutionForwardAlgorithm(m->handle.dnn, m->inputTensor, input_ptr,
                                                 m->filterDesc, filter_ptr, m->convDesc,
                                                 m->handle.workSpace, m->handle.workSpaceSize,
                                                 m->outputTensor, output_ptr);
  m->bwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
  m->bwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  if (m->relu) {
    checkCUDNN(cudnnCreateActivationDescriptor(&m->actiDesc));
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
  }
  return m;
}

__host__
void Conv2D::init(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<3> rect = runtime->get_index_space_domain(ctx, model.part_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    CnnHandle handle = model.cnn_handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(CnnHandle)));
  }
  IndexLauncher init_launcher(CONV2D_INIT_TASK_ID, model.part_is,
                              TaskArgument(this, sizeof(Conv2D)), argmap);
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
  for (PointInRectIterator<3> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I): filter
  regions[3](I): bias
*/
__host__
void Conv2D::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f, beta = 0.0f;
  const Conv2DMeta* m = *((Conv2DMeta**) task->local_args);
  const AccessorRO<float, 3> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 3> acc_output(regions[1], FID_DATA);
  const AccessorRO<float, 1> acc_filter(regions[2], FID_DATA);
  const AccessorRO<float, 1> acc_bias(regions[3], FID_DATA);
  Rect<3> rect_input, rect_output;
  Rect<1> rect_filter, rect_bias;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_filter = runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  for (int i = 0; i < 3; i++) printf("rect_input.hi = %lld lo = %lld\n", rect_input.hi[i], rect_input.lo[i]);
  for (int i = 0; i < 3; i++) printf("rect_output.hi = %lld lo = %lld\n", rect_output.hi[i], rect_output.lo[i]);
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_filter.accessor.is_dense_arbitrary(rect_filter));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);
  const float *filter_ptr = acc_filter.ptr(rect_filter.lo);
  const float *bias_ptr = acc_bias.ptr(rect_bias.lo);  

  printf("fwdAlgo(%d), bwdFilterALgo(%d), bwdDataAlgo(%d)\n", (int)m->fwdAlgo,(int) m->bwdFilterAlgo,(int) m->bwdDataAlgo);
  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start);
  double cp_1 = Realm::Clock::current_time_in_microseconds();
  checkCUDNN(cudnnConvolutionForward(m->handle.dnn, &alpha,
                                     m->inputTensor, input_ptr,
                                     m->filterDesc, filter_ptr,
                                     m->convDesc, m->fwdAlgo,
                                     m->handle.workSpace, m->handle.workSpaceSize,
                                     &beta, m->outputTensor, output_ptr));

  checkCUDNN(cudnnAddTensor(m->handle.dnn, &alpha, m->biasTensor,
                            bias_ptr, &alpha, m->outputTensor, output_ptr));
  double cp_2 = Realm::Clock::current_time_in_microseconds();
  if (m->relu) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
                                      &alpha, m->outputTensor, output_ptr,
                                      &beta, m->outputTensor, output_ptr));
  }
  cudaEventRecord(t_end);
  checkCUDA(cudaEventSynchronize(t_end));
  cudaDeviceSynchronize();
  double cp_3 = Realm::Clock::current_time_in_microseconds();
  float elapsed = 0;
  checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);

  printf("Conv2D forward time (CF) = %.2fus elapsed(%.2fms)\n", cp_2 - cp_1, elapsed);
  printf("Conv2D forward time (AF) = %.2fus\n", cp_3 - cp_2);
}

__host__
void Conv2D::forward(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  printf("CP#1\n");
  Rect<3> rect = runtime->get_index_space_domain(ctx, model.part_is);
  printf("CP#2\n");
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    printf("mp.pointer = %llx\n", mp);
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(CONV2D_FWD_TASK_ID, model.part_is,
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
void Conv2D::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f, beta = 0.0f;
}

__host__
void Conv2D::backward(const CnnModel& model)
{
}

bool checkConvolutionForwardAlgorithm(cudnnConvolutionFwdAlgo_t algo,
                                      int kernel_h, int kernel_w,
                                      int pad_h, int pad_w, int str_h, int str_w)
{
  switch (algo) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
    //case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMPUTED_GEMM:
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
      return true;
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
      return false;
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
      if ((str_h != 1)||(str_w != 1)) return false;
      if ((kernel_h < pad_h)||(kernel_w < pad_w)) return false;
      return true;
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
      if ((kernel_h > 32)||(kernel_w > 32)) return false;
      if ((str_h != 1)||(str_w != 1)) return false;
      if ((kernel_h < pad_h)||(kernel_w < pad_w)) return false;
      return true;
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
      if ((str_h != 1)||(str_w != 1)) return false;
      if ((kernel_h != 3)||(kernel_w != 3)) return false;
      return true;
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
      if ((str_h != 1)||(str_w != 1)) return false;
      if ((kernel_h == 3)&&(kernel_w == 3)) return true;
      if ((kernel_h == 5)&&(kernel_w == 5)) return true;
      return false;
    default:
      assert(0);
  }
  return false;
}

const int fwdAlgoCnt = 7;
const cudnnConvolutionFwdAlgo_t fwdAlgo[fwdAlgoCnt] = {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 
    //CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMPUTED_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};

cudnnConvolutionFwdAlgo_t
selectConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                  const cudnnTensorDescriptor_t xDesc, const void* x,
                                  const cudnnFilterDescriptor_t wDesc, const void* w,
                                  const cudnnConvolutionDescriptor_t convDesc,
                                  void* workSpace, size_t workSpaceSize,
                                  const cudnnTensorDescriptor_t yDesc, void* y)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  cudnnConvolutionFwdAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(
      handle, xDesc, x, wDesc, w, convDesc, yDesc, y,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("algo(%d) time(%.2lf)\n", perfResults[0].algo, perfResults[0].time);
  //for (int i = 0; i < cnt; i++) {
  //  printf("algo[%d] : algo(%d) time(%.2lf)\n", i, (int)perfResults[i].algo, perfResults[i].time);
  //};
  return perfResults[0].algo;
#ifdef DEADCODE
  long long cp_1, cp_2, best_time = LLONG_MAX;
  float alpha = 1.0f, beta = 0.0f, times[fwdAlgoCnt];
  cudnnConvolutionFwdAlgo_t best_algo;
  int pad_h, pad_w, str_h, str_w, upscale_x, upscale_y;
  cudnnConvolutionMode_t mode;
  int out_c, in_c, kernel_h, kernel_w;
  cudnnDataType_t type;
  cudnnTensorFormat_t format;
  checkCUDNN(cudnnGetConvolution2dDescriptor(convDesc, &pad_h, &pad_w, &str_h, &str_w,
                                             &upscale_x, &upscale_y, &mode));
  checkCUDNN(cudnnGetFilter4dDescriptor(wDesc, &type, &format, &out_c, &in_c, &kernel_h, &kernel_w));
  for (int i = 0; i < fwdAlgoCnt; i++) {
    if (!checkConvolutionForwardAlgorithm(fwdAlgo[i], kernel_h, kernel_w, pad_h, pad_w, str_h, str_w))
      continue;
    size_t size;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc,
                                                       convDesc, yDesc, fwdAlgo[i], &size));
    if (size > workSpaceSize) continue;
    //checkCUDA(cudaDeviceSynchronize());
    cudaEvent_t t_start, t_end;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, 0);
    cp_1 = Realm::Clock::current_time_in_microseconds();
    checkCUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, x, wDesc, w,
                                       convDesc, fwdAlgo[i], workSpace, workSpaceSize,
                                       &beta, yDesc, y));
    //checkCUDA(cudaDeviceSynchronize());
    cudaEventRecord(t_end, 0);
    float elapsed;
    checkCUDA(cudaEventSynchronize(t_end));
    cudaEventElapsedTime(&elapsed, t_start, t_end);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end); 
    cp_2 = Realm::Clock::current_time_in_microseconds();
    //printf("fwdAlgo[%d]: c_in(%d) c_out(%d) kernel(%d %d) stride(%d %d) time(%lld)\n", i, in_c, out_c, kernel_h, kernel_w, str_h, str_w, cp_2 - cp_1);
    //printf("Elapsed = %.2lfms\n", elapsed);
    times[i] = elapsed;
    if ((i == 0) || (cp_2 - cp_1 < best_time)) {
      best_time = cp_2 - cp_1;
      best_algo = fwdAlgo[i];
    }
  }
  for (int i = 0; i < fwdAlgoCnt; i++)
    printf("times[i] = %.2lfms\n", i, times[i]);
  printf("bestAlgo = %d\n", (int)best_algo);
  return best_algo;
#endif
}
