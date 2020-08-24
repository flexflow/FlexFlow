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

Tensor FFModel::conv2d(const Tensor& input,
                       int outChannels,
                       int kernelH, int kernelW,
                       int strideH, int strideW,
                       int paddingH, int paddingW,
                       ActiMode activation,
                       bool use_bias,
                       Initializer* kernel_initializer,
                       Initializer* bias_initializer)
{
  if (kernel_initializer == NULL) {
    int seed = std::rand();
    kernel_initializer = new GlorotUniform(seed);
  }
  if (bias_initializer == NULL) {
    bias_initializer = new ZeroInitializer();
  }

  assert(input.numDim == 4); /*NCHW*/
  Conv2D *conv = new Conv2D(*this, input, outChannels, kernelH, kernelW,
                            strideH, strideW, paddingH, paddingW, activation,
                            use_bias, kernel_initializer, bias_initializer);
  layers.push_back(conv);
  return conv->outputs[0];
}

Conv2D* FFModel::conv2d(int inChannels,
                        int outChannels,
                        int kernelH, int kernelW,
                        int strideH, int strideW,
                        int paddingH, int paddingW,
                        ActiMode activation,
                        bool use_bias,
                        Initializer* kernel_initializer,
                        Initializer* bias_initializer)
{
  if (kernel_initializer == NULL) {
    int seed = std::rand();
    kernel_initializer = new GlorotUniform(seed);
  }
  if (bias_initializer == NULL) {
    bias_initializer = new ZeroInitializer();
  }

  Conv2D *conv = new Conv2D(*this, inChannels, outChannels, kernelH, kernelW,
                            strideH, strideW, paddingH, paddingW, activation,
                            use_bias, kernel_initializer, bias_initializer);
  layers.push_back(conv);
  return conv;
}

/*
locals[0] = kernel
locals[1] = bias
*/
Conv2D::Conv2D(FFModel& model,
               const Tensor& _input,
               int out_dim,
               int _kernel_h, int _kernel_w,
               int _stride_h, int _stride_w,
               int _padding_h, int _padding_w,
               ActiMode _activation,
               bool _use_bias,
               Initializer* _kernel_initializer,
               Initializer* _bias_initializer)
: Op(model, OP_CONV2D, "Conv2D_"+std::to_string(_kernel_h)+std::to_string(_kernel_w), _input),
  in_channels(_input.adim[2]), out_channels(out_dim),
  kernel_h(_kernel_h), kernel_w(_kernel_w),
  stride_h(_stride_h), stride_w(_stride_w),
  padding_h(_padding_h), padding_w(_padding_w),
  activation(_activation), use_bias(_use_bias),
  kernel_initializer(_kernel_initializer),
  bias_initializer(_bias_initializer),
  profiling(model.config.profiling)
{
  assert(_input.numDim == 4);
  // Set output shape
  int input_w = inputs[0].adim[0];
  int input_h = inputs[0].adim[1];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_c = out_channels;
  int output_n = inputs[0].adim[3];
  outputs[0].numDim = 4;
  outputs[0].adim[0] = output_w;
  outputs[0].adim[1] = output_h;
  outputs[0].adim[2] = output_c;
  outputs[0].adim[3] = output_n;
  weights[0].numDim = 4;
  weights[0].adim[0] = kernel_w;
  weights[0].adim[1] = kernel_h;
  weights[0].adim[2] = in_channels;
  weights[0].adim[3] = out_channels;
  numWeights = 1;
  if (use_bias) {
    weights[1].numDim = 1;
    weights[1].adim[0] = out_channels;
    numWeights = 2;
  }
}

Conv2D::Conv2D(FFModel& model,
               int in_dim, int out_dim,
               int _kernel_h, int _kernel_w,
               int _stride_h, int _stride_w,
               int _padding_h, int _padding_w,
               ActiMode _activation,
               bool _use_bias,
               Initializer* _kernel_initializer,
               Initializer* _bias_initializer)
: Op(model, OP_CONV2D, "Conv2D_"+std::to_string(_kernel_h)+std::to_string(_kernel_w), 1),
  in_channels(in_dim), out_channels(out_dim),
  kernel_h(_kernel_h), kernel_w(_kernel_w),
  stride_h(_stride_h), stride_w(_stride_w),
  padding_h(_padding_h), padding_w(_padding_w),
  activation(_activation), use_bias(_use_bias),
  kernel_initializer(_kernel_initializer),
  bias_initializer(_bias_initializer),
  profiling(model.config.profiling)
{
}

Tensor Conv2D::init_inout(FFModel& model, const Tensor& _input)
{
  assert(_input.numDim == 4);
  assert(_input.adim[2] == in_channels);
  inputs[0] = _input;
  create_output_and_partition(model);
  return outputs[0];
}

void Conv2D::create_weights(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<4>(model.get_or_create_task_is(4, pcname));
  
  // Create kernel
  {
    const int dims[4] = {out_channels, in_channels, kernel_h, kernel_w};
    weights[0] = model.create_conv_weight<4>(this, dims, (IndexSpaceT<4>)task_is, DT_FLOAT, kernel_initializer);
  }
  // Create bias tensor
  if (use_bias) {
    const int dims[1] = {out_channels};
    weights[1] = model.create_conv_weight<1>(this, dims, (IndexSpaceT<4>)task_is, DT_FLOAT, bias_initializer);
    assert(numWeights == 2);
  } else {
    assert(numWeights == 1);
  }
}

void Conv2D::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<4>(model.get_or_create_task_is(4, pcname));

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // Create output tensor
  int input_w = inputs[0].adim[0];
  int input_h = inputs[0].adim[1];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_c = out_channels;
  int output_n = inputs[0].adim[3];
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;
  {
    const int dims[4] = {output_n, output_c, output_h, output_w};
    outputs[0] = model.create_tensor<4>(dims, (IndexSpaceT<4>)task_is, DT_FLOAT);
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }
  // Compute partition bound for input
  Rect<4> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
  // Currently assume we didn't split across the channel dimension
  assert(num_par_c == 1);
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0].part;
    input_grad_lps[0] = inputs[0].part_grad;
  } else {
    model.create_disjoint_partition(
        inputs[0], (IndexSpaceT<4>)task_is, input_lps[0], input_grad_lps[0]);
  }
}

cudnnConvolutionFwdAlgo_t
selectConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                  const cudnnTensorDescriptor_t xDesc, const void* x,
                                  const cudnnFilterDescriptor_t wDesc, const void* w,
                                  const cudnnConvolutionDescriptor_t convDesc,
                                  void* workSpace, size_t workSpaceSize,
                                  const cudnnTensorDescriptor_t yDesc, void* y);
cudnnConvolutionBwdFilterAlgo_t
selectConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                         const cudnnTensorDescriptor_t xDesc, const void* x,
                                         const cudnnTensorDescriptor_t dyDesc, const void* dy,
                                         const cudnnConvolutionDescriptor_t convDesc,
                                         void* workSpace, size_t workSpaceSize,
                                         const cudnnFilterDescriptor_t dwDesc, void* dw);
cudnnConvolutionBwdDataAlgo_t
selectConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                       const cudnnFilterDescriptor_t wDesc, const void* w,
                                       const cudnnTensorDescriptor_t dyDesc, const void* dy,
                                       const cudnnConvolutionDescriptor_t convDesc,
                                       void* workSpace, size_t workSpaceSize,
                                       const cudnnTensorDescriptor_t dxDesc, void* dx);
/*
  regions[0]: input
  regions[1]: output
  regions[2](I): filter
  regions[3](I): bias
  regions[4](O): filter_grad
  regions[5](O): input_grad
*/
__host__
OpMeta* Conv2D::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 6);
  assert(task->regions.size() == 6);
  const Conv2D* conv = (Conv2D*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, 4> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 1> acc_bias(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_kernel_grad(
      regions[4], task->regions[4], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorW<float, 4> acc_input_grad(
      regions[5], task->regions[5], FID_DATA, ctx, runtime,
      false/*readOutput*/);
 
  Conv2DMeta* m = new Conv2DMeta(handle);
  m->relu = conv->activation == AC_MODE_RELU;

  int input_w = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int input_h = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
  int input_c = acc_input.rect.hi[2] - acc_input.rect.lo[2] + 1;
  int input_n = acc_input.rect.hi[3] - acc_input.rect.lo[3] + 1;
  int output_w = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int output_h = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
  int output_c = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
  int output_n = acc_output.rect.hi[3] - acc_output.rect.lo[3] + 1;
  printf("init conv (input): n(%d) c(%d) h(%d) w(%d)\n",
         input_n, input_c, input_h, input_w);
  printf("init conv (output): n(%d) c(%d) h(%d) w(%d)\n",
          output_n, output_c, output_h, output_w);
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
      CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      input_n, input_c, input_h, input_w));
  
  checkCUDNN(cudnnSetTensor4dDescriptor(m->biasTensor,
      CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      1, output_c, 1, 1));

  printf("filterDim: kernel(%d %d) c_in(%d), c_out(%d)\n", conv->kernel_h, conv->kernel_w, input_c, output_c);
  checkCUDNN(cudnnSetFilter4dDescriptor(m->filterDesc,
      CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
      output_c, input_c, conv->kernel_h, conv->kernel_w));

  //printf("convDim: padding(%d %d) stride(%d %d)\n", conv->padding_h, conv->padding_w, conv->stride_h, conv->stride_w);
  int pad_h = ((output_h - 1) * conv->stride_h + conv->kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * conv->stride_w + conv->kernel_w - input_w + 1) / 2;
  if (pad_h != conv->padding_h)
    printf("Warning: changing conv_padding_h to satisfy output_h size\n");
  if (pad_w != conv->padding_w)
    printf("Warning: changing conv_padding_w to satisfy output_w size\n");

  checkCUDNN(cudnnSetConvolution2dDescriptor(m->convDesc,
                                             pad_h,//conv->padding_h,
                                             pad_w,//conv->padding_w,
                                             conv->stride_h,
                                             conv->stride_w,
                                             1/*upscale_x*/,
                                             1/*upscale_y*/,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));
  // enable tensor core when possible
  checkCUDNN(cudnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH));
  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(m->convDesc,
                                                   m->inputTensor,
                                                   m->filterDesc,
                                                   &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  // select forward algorithm
  m->fwdAlgo = selectConvolutionForwardAlgorithm(m->handle.dnn, m->inputTensor, acc_input.ptr,
                                                 m->filterDesc, acc_kernel.ptr, m->convDesc,
                                                 m->handle.workSpace, m->handle.workSpaceSize,
                                                 m->outputTensor, acc_output.ptr);
  // select backward filter algorithm
  m->bwdFilterAlgo = selectConvolutionBackwardFilterAlgorithm(
                         m->handle.dnn, m->inputTensor, acc_input.ptr,
                         m->outputTensor, acc_output.ptr,
                         m->convDesc, m->handle.workSpace, m->handle.workSpaceSize,
                         m->filterDesc, acc_kernel_grad.ptr);
  // select backward data algorithm
  m->bwdDataAlgo = selectConvolutionBackwardDataAlgorithm(
                       m->handle.dnn, m->filterDesc, acc_kernel.ptr,
                       m->outputTensor, acc_output.ptr,
                       m->convDesc, m->handle.workSpace, m->handle.workSpaceSize,
                       m->inputTensor, acc_input_grad.ptr);
  if (m->relu) {
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
  }
  return m;
}

void Conv2D::init(const FFModel& ff)
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
  IndexLauncher launcher(CONV2D_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Conv2D)), argmap,
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
  launcher.add_region_requirement(
      RegionRequirement(weights[0].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0].region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[1].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[1].region));
  launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0].part_grad, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, weights[0].region_grad));
  launcher.add_field(4, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(5, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

void Conv2D::forward_kernel(const Conv2DMeta* m,
                            const float* input_ptr,
                            float* output_ptr,
                            const float* filter_ptr,
                            const float* bias_ptr)
{
  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(m->handle.dnn, &alpha,
                                     m->inputTensor, input_ptr,
                                     m->filterDesc, filter_ptr,
                                     m->convDesc, m->fwdAlgo,
                                     m->handle.workSpace, m->handle.workSpaceSize,
                                     &beta, m->outputTensor, output_ptr));

  checkCUDNN(cudnnAddTensor(m->handle.dnn, &alpha, m->biasTensor,
                            bias_ptr, &alpha, m->outputTensor, output_ptr));
  if (m->relu) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
                                      &alpha, m->outputTensor, output_ptr,
                                      &beta, m->outputTensor, output_ptr));
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
  Conv2D* conv = (Conv2D*) task->args;
  const Conv2DMeta* m = *((Conv2DMeta**) task->local_args);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, 4> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 1> acc_bias(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);

  //printf("fwdAlgo(%d), bwdFilterALgo(%d), bwdDataAlgo(%d)\n", (int)m->fwdAlgo,(int) m->bwdFilterAlgo,(int) m->bwdDataAlgo);
  cudaEvent_t t_start, t_end;
  if (conv->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }

#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  conv->forward_kernel(m, acc_input.ptr, acc_output.ptr, acc_kernel.ptr, acc_bias.ptr);
  if (conv->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    //print_tensor<4, float>(acc_input.ptr, acc_input.rect, "[Conv2D:forward:input]");
    //print_tensor<4, float>(acc_kernel.ptr, acc_kernel.rect, "[Conv2D:forward:kernel]");
    //print_tensor<1, float>(acc_bias.ptr, acc_bias.rect, "[Conv2D:forward:bias]");
    //print_tensor<4, float>(acc_output.ptr, acc_output.rect, "[Conv2D:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Conv2D forward time (CF) = %.2fms\n", elapsed);
  }
}

__host__
void Conv2D::forward(const FFModel& ff)
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
  IndexLauncher launcher(CONV2D_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Conv2D)), argmap,
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
  launcher.add_region_requirement(
      RegionRequirement(weights[0].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0].region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[1].region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[1].region));
  launcher.add_field(3, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Conv2D::backward_kernel(const Conv2DMeta* m,
                             const float* input_ptr,
                             float* input_grad_ptr,
                             const float* output_ptr,
                             float* output_grad_ptr,
                             const float* kernel_ptr,
                             float* kernel_grad_ptr,
                             float* bias_grad_ptr)
{
  float alpha = 1.0f;
  //float beta = 0.0f;
  if (m->relu) {
    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    checkCUDNN(cudnnGetTensor4dDescriptor(m->outputTensor, &dataType,
        &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));
    reluBackward<<<GET_BLOCKS(n*c*h*w), CUDA_NUM_THREADS>>>(output_grad_ptr, output_ptr, n*c*h*w);
  }
  // Compute filter gradiant
  // NOTE: we use alpha for kernel_grad to accumulate gradients
  checkCUDNN(cudnnConvolutionBackwardFilter(m->handle.dnn, &alpha,
                                            m->inputTensor, input_ptr,
                                            m->outputTensor, output_grad_ptr,
                                            m->convDesc, m->bwdFilterAlgo,
                                            m->handle.workSpace, m->handle.workSpaceSize,
                                            &alpha, m->filterDesc, kernel_grad_ptr));
  // Compute bias gradiant
  // NOTE: we use alpha for bias_grad to accumulate gradients
  checkCUDNN(cudnnConvolutionBackwardBias(m->handle.dnn, &alpha,
                                          m->outputTensor, output_grad_ptr,
                                          &alpha, m->biasTensor, bias_grad_ptr));
  // Compute data gradiant
  // NOTE: we use alpha for input_grad to accumulate gradients
  checkCUDNN(cudnnConvolutionBackwardData(m->handle.dnn, &alpha,
                                          m->filterDesc, kernel_ptr,
                                          m->outputTensor, output_grad_ptr,
                                          m->convDesc, m->bwdDataAlgo,
                                          m->handle.workSpace, m->handle.workSpaceSize,
                                          &alpha, m->inputTensor, input_grad_ptr));
}

/*
  regions[0](I): input
  regions[1](I/O): input_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): filter
  regions[5](I/O): filter_grad
  regions[6](I/O): bias_grad
*/
__host__
void Conv2D::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);
  Conv2D* conv = (Conv2D*) task->args;
  const Conv2DMeta* m = *((Conv2DMeta**) task->local_args);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_input_grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, 4> acc_output(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_output_grad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime,
      true/*rreadOutput*/);
  TensorAccessorR<float, 4> acc_kernel(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_kernel_grad(
      regions[5], task->regions[5], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorW<float, 1> acc_bias_grad(
      regions[6], task->regions[6], FID_DATA, ctx, runtime,
      true/*readOutput*/);

  cudaEvent_t t_start, t_end;
  if (conv->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }

#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  conv->backward_kernel(m, acc_input.ptr, acc_input_grad.ptr,
                        acc_output.ptr, acc_output_grad.ptr,
                        acc_kernel.ptr, acc_kernel_grad.ptr,
                        acc_bias_grad.ptr);
  if (conv->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Conv2D backward time = %.2fms\n", elapsed);
    //print_tensor<4, float>(acc_output_grad.ptr, acc_output_grad.rect, "[Conv2D:backward:output_grad]");
    //print_tensor<4, float>(acc_kernel_grad.ptr, acc_kernel_grad.rect, "[Conv2D:backward:kernel_grad]");
    //print_tensor<1, float>(acc_bias_grad.ptr, acc_bias_grad.rect, "[Conv2D:backward:bias_grad]");
    //print_tensor<4, float>(acc_input_grad.ptr, acc_input_grad.rect, "[Conv2D:backward:input_grad]");
  }
}

__host__
void Conv2D::backward(const FFModel& ff)
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

  IndexLauncher launcher(CONV2D_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Conv2D)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0](I): input
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
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
  // regions[3](I/O): output_grad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(3, FID_DATA);
  // regions[4](I): filter
  launcher.add_region_requirement(
      RegionRequirement(weights[0].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0].region));
  launcher.add_field(4, FID_DATA);
  // regions[5](I/O): filter_grad
  launcher.add_region_requirement(
      RegionRequirement(weights[0].part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, weights[0].region_grad));
  launcher.add_field(5, FID_DATA);
  // regions[6](I/O): bias_grad
  launcher.add_region_requirement(
      RegionRequirement(weights[1].part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, weights[1].region_grad));
  launcher.add_field(6, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  // TODO: remove this line
  //if (first_layer)
    //fm.wait_all_results();
}

#ifdef DEADCODE
/*
  regions[0](I/O): filter
  regions[1](I): filter_grad
  regions[2](I/O): bias
  regions[3](I): bias_grad
*/
__host__
void Conv2D::update_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const Conv2D* conv = (Conv2D*) task->args;
  const AccessorRW<float, 1> acc_filter(regions[0], FID_DATA);
  const AccessorRO<float, 1> acc_filter_grad(regions[1], FID_DATA);
  const AccessorRW<float, 1> acc_bias(regions[2], FID_DATA);
  const AccessorRO<float, 1> acc_bias_grad(regions[3], FID_DATA);
  Rect<1> rect_filter, rect_filter_grad, rect_bias, rect_bias_grad;
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
  assert(filter_size == conv->in_channels * conv->out_channels
                        * conv->kernel_w * conv->kernel_h);
  assert(bias_size == conv->out_channels);
  assert(filter_size * conv->num_replica == rect_filter_grad.volume());
  assert(bias_size * conv->num_replica == rect_bias_grad.volume());
  assert(acc_filter.accessor.is_dense_arbitrary(rect_filter));
  assert(acc_filter_grad.accessor.is_dense_arbitrary(rect_filter_grad));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  assert(acc_bias_grad.accessor.is_dense_arbitrary(rect_bias_grad));
  float *filter_ptr = acc_filter.ptr(rect_filter.lo);
  const float *filter_grad_ptr = acc_filter_grad.ptr(rect_filter_grad.lo);
  float *bias_ptr = acc_bias.ptr(rect_bias.lo);
  const float *bias_grad_ptr = acc_bias_grad.ptr(rect_bias_grad.lo);
  updateGAS(filter_ptr, filter_grad_ptr, filter_size,
            conv->num_replica, conv->learning_rate);
  updateGAS(bias_ptr, bias_grad_ptr, bias_size,
            conv->num_replica, conv->learning_rate);
}

__host__
void Conv2D::update(const FFModel& ff)
{
  // Synchronize the learning rate
  learning_rate = ff.config.learningRate;
  assert(num_replica > 0);
  // Only aggregate parameters if more than one replica
  if (num_replica > 1) {
    Context ctx = ff.config.lg_ctx;
    Runtime* runtime = ff.config.lg_hlr;
    TaskLauncher launcher(CONV2D_UPD_TASK_ID, TaskArgument(this, sizeof(Conv2D)));
    launcher.add_region_requirement(
      RegionRequirement(locals[0].region, READ_WRITE, EXCLUSIVE, locals[0].region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(locals[0].region_grad, READ_ONLY, EXCLUSIVE, locals[0].region_grad));
    launcher.add_field(1, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(locals[1].region, READ_WRITE, EXCLUSIVE, locals[1].region));
    launcher.add_field(2, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(locals[1].region_grad, READ_ONLY, EXCLUSIVE, locals[1].region_grad));
    launcher.add_field(3, FID_DATA);
    runtime->execute_task(ctx, launcher);
  }
}
#endif

/*
__host__
Parameter* Conv2D::get_parameter(int index)
{
  if (index == 0) {
    return &weights[0];
  } else if (index == 1) {
    return &weights[1];
  } else {
    assert(0);
    return NULL;
  }
}*/

__host__
void Conv2D::print_layer(const FFModel& ff)
{
  printf("conv2d layer\n");  
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
#if 0
  TaskLauncher launcher(CONV2D_PRINT_TASK_ID, TaskArgument(NULL, 0));
  launcher.add_region_requirement(
    RegionRequirement(kernel.region, READ_ONLY, EXCLUSIVE, kernel.region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(bias.region, READ_ONLY, EXCLUSIVE, bias.region));
  launcher.add_field(1, FID_DATA);
  Future fu = runtime->execute_task(ctx, launcher);
  fu.wait();
#else
  RegionRequirement kernel_req(weights[0].region, READ_WRITE, EXCLUSIVE, weights[0].region);
  kernel_req.add_field(FID_DATA);
  InlineLauncher kernel_launcher(kernel_req);
  PhysicalRegion kernel_region = runtime->map_region(ctx, kernel_launcher);
  kernel_region.wait_until_valid();

/*  
  RegionRequirement kernel_grad_req(kernel.region_grad, READ_WRITE, EXCLUSIVE, kernel.region_grad);
  kernel_grad_req.add_field(FID_DATA);
  InlineLauncher kernel_grad_launcher(kernel_grad_req);
  PhysicalRegion kernel_grad_region = runtime->map_region(ctx, kernel_grad_launcher);
  kernel_grad_region.wait_until_valid();
*/  
  RegionRequirement bias_req(weights[1].region, READ_WRITE, EXCLUSIVE, weights[1].region);
  bias_req.add_field(FID_DATA);
  InlineLauncher bias_launcher(bias_req);
  PhysicalRegion bias_region = runtime->map_region(ctx, bias_launcher);
  bias_region.wait_until_valid();
/*  
  RegionRequirement bias_grad_req(bias.region_grad, READ_WRITE, EXCLUSIVE, bias.region_grad);
  bias_grad_req.add_field(FID_DATA);
  InlineLauncher bias_grad_launcher(bias_grad_req);
  PhysicalRegion bias_grad_region = runtime->map_region(ctx, bias_grad_launcher);
  bias_grad_region.wait_until_valid();
  */
  TensorAccessorW<float, 4> acc_kernel(kernel_region, kernel_req, FID_DATA, ctx, runtime, true);
//  const AccessorRW<float, 1> acc_kernel_grad(kernel_grad_region, FID_DATA);
  TensorAccessorW<float, 1> acc_bias(bias_region, bias_req, FID_DATA, ctx, runtime, true);
  //const AccessorRW<float, 1> acc_bias_grad(bias_grad_region, FID_DATA);
  
  const float *kernel_ptr = acc_kernel.ptr;
  //float *kernel_grad_ptr = acc_kernel_grad.ptr;
  const float *bias_ptr = acc_bias.ptr;
  //float *bias_grad_ptr = acc_bias_grad.ptr;
  
  size_t kernel_size = acc_kernel.rect.volume();
  int kernel_dim1 = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int kernel_dim2 = acc_kernel.rect.hi[1] - acc_kernel.rect.lo[1] + 1;
  int kernel_dim3 = acc_kernel.rect.hi[2] - acc_kernel.rect.lo[2] + 1;
  int kernel_dim4 = acc_kernel.rect.hi[3] - acc_kernel.rect.lo[3] + 1;
  //size_t kernel_grad_size = rect_kernel_grad.volume();
  size_t bias_size = acc_bias.rect.volume();
  //size_t bias_grad_size = rect_bias_grad.volume();
  printf("kernel, %p, %d, [%d, %d, %d, %d]\n", kernel_ptr, kernel_size, kernel_dim1, kernel_dim2, kernel_dim3, kernel_dim4);
  //printf("kernel_grad, %d\n", kernel_grad_size);
  printf("bias, %p, %d\n", bias_ptr, bias_size);
  //printf("bias_grad, %d\n", bias_grad_size);

  
  for (int i = 0; i < bias_size; i++) {
    printf("%f ", bias_ptr[i]);
  }
  printf("\n");
  
/*  
  for (int i = 0; i < bias_grad_size; i++) {
    printf("%f ", bias_grad_ptr);
    bias_grad_ptr ++;
  }
  printf("\n");*/
  
  for (int i = 0; i < kernel_size; i++) {
    printf("%f ", kernel_ptr[i]);
  }
  printf("\n");
  
/*  
  for (int i = 0; i < kernel_grad_size; i++) {
    printf("%f ", kernel_grad_ptr);
    kernel_grad_ptr ++;
  }
  printf("\n");
  */
  runtime->unmap_region(ctx, kernel_region);
 // runtime->unmap_region(ctx, kernel_grad_region);
  runtime->unmap_region(ctx, bias_region);
//  runtime->unmap_region(ctx, bias_grad_region);
#endif
}

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
  printf("forwardAlgo(%d) time(%.2lf)\n", perfResults[0].algo, perfResults[0].time);
  return perfResults[0].algo;
}

cudnnConvolutionBwdFilterAlgo_t
selectConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                         const cudnnTensorDescriptor_t xDesc, const void* x,
                                         const cudnnTensorDescriptor_t dyDesc, const void* dy,
                                         const cudnnConvolutionDescriptor_t convDesc,
                                         void* workSpace, size_t workSpaceSize,
                                         const cudnnFilterDescriptor_t dwDesc, void* dw)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(
      handle, xDesc, x, dyDesc, dy, convDesc, dwDesc, dw,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("bwdFilterAlgo(%d) time(%.2lf)\n", perfResults[0].algo, perfResults[0].time);
  return perfResults[0].algo;
}

cudnnConvolutionBwdDataAlgo_t
selectConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                       const cudnnFilterDescriptor_t wDesc, const void* w,
                                       const cudnnTensorDescriptor_t dyDesc, const void* dy,
                                       const cudnnConvolutionDescriptor_t convDesc,
                                       void* workSpace, size_t workSpaceSize,
                                       const cudnnTensorDescriptor_t dxDesc, void* dx)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithmEx(
      handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("bwdDataAlgo(%d) time(%.2lf)\n", perfResults[0].algo, perfResults[0].time);
  return perfResults[0].algo;
}

Conv2DMeta::Conv2DMeta(FFHandler handler)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
}

bool Conv2D::measure_compute_time(Simulator* sim,
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

  Conv2DMeta* m = sim->conv2d_meta;
  m->relu = activation == AC_MODE_RELU;
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));
  checkCUDNN(cudnnSetTensor4dDescriptor(m->biasTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, 1, output_c, 1, 1));
  checkCUDNN(cudnnSetFilter4dDescriptor(m->filterDesc, CUDNN_DATA_FLOAT,
      CUDNN_TENSOR_NCHW, output_c, input_c, kernel_h, kernel_w));
  checkCUDNN(cudnnSetConvolution2dDescriptor(m->convDesc, pad_h, pad_w,
      stride_h, stride_w, 1/*dilationH*/, 1/*dilationW*/,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  checkCUDNN(cudnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH));
  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(m->convDesc,
      m->inputTensor, m->filterDesc, &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);
  checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, CUDNN_ACTIVATION_RELU,
      CUDNN_NOT_PROPAGATE_NAN, 0.0));
  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, n, c, h, w));
  // allocate tensors in simulator
  sim->free_all();
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  float *output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  float* weight_ptr = (float*)sim->allocate((size_t)output_c * input_c * kernel_h * kernel_w, DT_FLOAT);
  assert(weight_ptr != NULL);
  float* bias_ptr = (float*)sim->allocate(output_c, DT_FLOAT);
  assert(bias_ptr != NULL);
  
  // select forward algorithm
  {
    const int reqAlgCnt = 8;
    int cnt = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults[reqAlgCnt];
    checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(
        m->handle.dnn, m->inputTensor, input_ptr,
        m->filterDesc, weight_ptr, m->convDesc, m->outputTensor, output_ptr,
        reqAlgCnt, &cnt, perfResults,
        m->handle.workSpace, m->handle.workSpaceSize));
    assert(cnt > 0);
    checkCUDNN(perfResults[0].status);
    forward_time = perfResults[0].time;
    //for (int i = 0; i < cnt; i++)
    //  printf("conv forward: algo(%d) time(%.4lf)\n", perfResults[i].algo, perfResults[i].time);
  }
  // select forward algorithm
  {
    const int reqAlgCnt = 8;
    int cnt = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults[reqAlgCnt];
    checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        m->handle.dnn, m->inputTensor, input_ptr,
        m->outputTensor, output_ptr, m->convDesc, m->filterDesc, weight_ptr,
        reqAlgCnt, &cnt, perfResults,
        m->handle.workSpace, m->handle.workSpaceSize));
    assert(cnt > 0);
    checkCUDNN(perfResults[0].status);
    backward_time = perfResults[0].time;
  }
  {
    const int reqAlgCnt = 8;
    int cnt = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perfResults[reqAlgCnt];
    checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithmEx(
        m->handle.dnn, m->filterDesc, weight_ptr,
        m->outputTensor, output_ptr, m->convDesc, m->inputTensor, input_ptr,
        reqAlgCnt, &cnt, perfResults,
        m->handle.workSpace, m->handle.workSpaceSize));
    assert(cnt > 0);
    checkCUDNN(perfResults[0].status);
    backward_time += perfResults[0].time;
  }
  printf("[Measure Conv2D] input(%d %d %d %d) output(%d %d %d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
         input_n, input_c, input_h, input_w, output_n, output_c, output_h, output_w,
         forward_time, backward_time);
  return true;
}

