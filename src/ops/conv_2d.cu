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

Tensor FFModel::conv2d(std::string name,
                       const Tensor& input,
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
    //kernel_initializer = new GlorotUniform(seed);
    kernel_initializer = new ZeroInitializer();
  }
  if (bias_initializer == NULL) {
    bias_initializer = new ZeroInitializer();
  }

  assert(input.numDim == 4); /*NCHW*/
  Conv2D *conv = new Conv2D(*this, name, input, outChannels, kernelH, kernelW,
                            strideH, strideW, paddingH, paddingW, activation,
                            use_bias, kernel_initializer, bias_initializer);
  layers.push_back(conv);
  Parameter kernel, bias;
  kernel.tensor = conv->kernel;
  kernel.op = conv;
  parameters.push_back(kernel);
  if (use_bias) {
    bias.tensor = conv->bias;
    bias.op = conv;
    parameters.push_back(bias);
  }
  return conv->output;
}

Conv2D* FFModel::conv2d(std::string name,
                       int inChannels,
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
    //kernel_initializer = new GlorotUniform(seed);
    kernel_initializer = new ZeroInitializer();
  }
  if (bias_initializer == NULL) {
    bias_initializer = new ZeroInitializer();
  }

  Conv2D *conv = new Conv2D(*this, name, inChannels, outChannels, kernelH, kernelW,
                            strideH, strideW, paddingH, paddingW, activation,
                            use_bias, kernel_initializer, bias_initializer);
  return conv;
}

/*
locals[0] = kernel
locals[1] = bias
*/
Conv2D::Conv2D(FFModel& model,
               const std::string& pcname,
               const Tensor& _input,
               int out_dim,
               int _kernel_h, int _kernel_w,
               int _stride_h, int _stride_w,
               int _padding_h, int _padding_w,
               ActiMode _activation,
               bool use_bias,
               Initializer* kernel_initializer,
               Initializer* bias_initializer)
: Op(pcname, _input),
  in_channels(_input.adim[2]), out_channels(out_dim),
  kernel_h(_kernel_h), kernel_w(_kernel_w),
  stride_h(_stride_h), stride_w(_stride_w),
  padding_h(_padding_h), padding_w(_padding_w),
  activation(_activation), profiling(model.config.profiling)
{
  assert(_input.numDim == 4);
    // Retrive the task indexspace for the op
  task_is = IndexSpaceT<4>(model.get_or_create_task_is(4, pcname));

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // Create output tensor
  int input_w = _input.adim[0];
  int input_h = _input.adim[1];
  int input_c = _input.adim[2];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_c = out_dim;
  int output_n = _input.adim[3];
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;
  {
    const int dims[4] = {output_n, output_c, output_h, output_w};
    output = model.create_tensor<4>(dims, task_is, DT_FLOAT);
  }
  // Create kernel
  {
    const int dims[4] = {output_c, input_c, kernel_h, kernel_w};
    kernel = model.create_conv_weight<4>(dims, task_is, DT_FLOAT, kernel_initializer);
    //printf("kernel ndim %d, [%d, %d, %d, %d]\n", kernel.numDim, kernel.adim[0], kernel.adim[1], kernel.adim[2], kernel.adim[3]);
  }
  // Create bias tensor
  if (use_bias) {
    const int dims[1] = {output_c};
    bias = model.create_conv_weight<1>(dims, task_is, DT_FLOAT, bias_initializer);
  //  printf("bias ndim %d, [%d, %d, %d, %d]\n", bias.numDim, bias.adim[0], bias.adim[1], bias.adim[2], bias.adim[3]);
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
        inputs[0], task_is, input_lps[0], input_grad_lps[0]);
  }
#ifdef DEADCODE
  IndexSpaceT<4> output_is;
  {
    //const Legion::coord_t lo[4] = {0, 0, 0, 0};
    //const Legion::coord_t hi[4] = {output_w-1, output_h-1, output_c-1, output_n-1};
    Rect<4> output_rect(Point<4>(0, 0, 0, 0),
        Point<4>(output_w-1, output_h-1, output_c-1, output_n-1));
    output_is = runtime->create_index_space<4>(ctx, output_rect);
  }
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  LogicalRegion output_grad_lr = runtime->create_logical_region(ctx, output_is, fs);
  int extent_w = (output_w + num_par_w - 1) / num_par_w;
  int extent_h = (output_h + num_par_h - 1) / num_par_h;
  int extent_c = output_c / num_par_c;
  int extent_n = output_n / num_par_n;
  assert(output_c % num_par_c == 0);
  assert(output_n % num_par_n == 0);
  Transform<4, 4, coord_t> transform;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      transform[i][j] = 0;
  transform[0][0] = extent_w;
  transform[1][1] = extent_h;
  transform[2][2] = extent_c;
  transform[3][3] = extent_n;
  IndexPartition output_ip;
  {
    //int lo[4] = {0, 0, 0, 0};
    //int hi[4] = {extent_w-1, extent_h-1, extent_c-1, extent_n-1};
    Rect<4> extent(Realm::Point<4>(0, 0, 0, 0),
        Realm::Point<4>(extent_w-1, extent_h-1, extent_c-1, extent_n-1));
    output_ip = runtime->create_partition_by_restriction(ctx, output_is, task_is, transform, extent);
    assert(runtime->is_index_partition_disjoint(ctx, output_ip));
    assert(runtime->is_index_partition_complete(ctx, output_ip));
  }
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  LogicalPartition output_grad_lp =
    runtime->get_logical_partition(ctx, output_grad_lr, output_ip);

  int kernel_nc = num_replica * in_channels * out_channels;
  Rect<1, coord_t> kernel_rect(0, kernel_w * kernel_h * in_channels * out_channels - 1);
  Rect<1, coord_t> kernel_grad_rect(0, kernel_w * kernel_h * kernel_nc - 1);
  IndexSpaceT<1> kernel_is = runtime->create_index_space(ctx, kernel_rect);
  IndexSpaceT<1> kernel_grad_is = runtime->create_index_space(ctx, kernel_grad_rect);
  LogicalRegion kernel_lr = runtime->create_logical_region(ctx, kernel_is, fs);
  LogicalRegion kernel_grad_lr = runtime->create_logical_region(ctx, kernel_grad_is, fs);
  IndexPartition kernel_grad_ip =
    runtime->create_equal_partition(ctx, kernel_grad_is, task_is);
  LogicalPartition kernel_grad_lp =
    runtime->get_logical_partition(ctx, kernel_grad_lr, kernel_grad_ip);
  Tensor kernel_tensor;
  kernel_tensor.numDim = 0;
  kernel_tensor.region = kernel_lr;
  kernel_tensor.region_grad = kernel_grad_lr;
  kernel_tensor.part = LogicalPartition::NO_PART;
  kernel_tensor.part_grad = kernel_grad_lp;
  locals[0] = kernel_tensor;

  int bias_nc = num_replica * out_channels;
  Rect<1, coord_t> bias_grad_rect(0, bias_nc - 1);
  Rect<1, coord_t> bias_rect(0, out_channels - 1);
  IndexSpaceT<1> bias_is = runtime->create_index_space(ctx, bias_rect);
  IndexSpaceT<1> bias_grad_is = runtime->create_index_space(ctx, bias_grad_rect);
  LogicalRegion bias_lr = runtime->create_logical_region(ctx, bias_is, fs);
  LogicalRegion bias_grad_lr =
    runtime->create_logical_region(ctx, bias_grad_is, fs);
  IndexPartition bias_grad_ip =
    runtime->create_equal_partition(ctx, bias_grad_is, task_is);
  LogicalPartition bias_grad_lp =
    runtime->get_logical_partition(ctx, bias_grad_lr, bias_grad_ip);
  Tensor bias_tensor;
  bias_tensor.numDim = 0;
  bias_tensor.region = bias_lr;
  bias_tensor.region_grad = bias_grad_lr;
  bias_tensor.part = LogicalPartition::NO_PART;
  bias_tensor.part_grad = bias_grad_lp;
  locals[1] = bias_tensor;
  numLocals = 2;

  output.numDim = 4;
  output.adim[0] = output_w;
  output.adim[1] = output_h;
  output.adim[2] = out_channels;
  output.adim[3] = _input.adim[3];
  output.pdim[0] = extent_w;
  output.pdim[1] = extent_h;
  output.pdim[2] = extent_c;
  output.pdim[3] = extent_n;
  output.region = output_lr;
  output.part = output_lp;
  output.region_grad = output_grad_lr;
  output.part_grad = output_grad_lp;
  printf("Create conv layer: name %s, output(n=%d c=%d h=%d w=%d)\n",
         pcname.c_str(), output.adim[3], output.adim[2], output.adim[1], output.adim[0]);

  // Compute partition bound for input
  Rect<4> input_part_rect =
    runtime->get_index_partition_color_space(ctx, inputs[0].part.get_index_partition());
  if (input_part_rect == part_rect) {
    input_lps[0] = _input.part;
  } else {
    printf("WARNING: input has a different partition!!!\n");
    IndexSpaceT<4> input_is = IndexSpaceT<4>(inputs[0].region.get_index_space());
    //extent_w = stride_w * (output.pdim[0]-1) + kernel_w - 2 * padding_w;
    //extent_h = stride_h * (output.pdim[1]-1) + kernel_h - 2 * padding_h;
    //extent_nc = inputs[0].adim[2] * inputs[0].adim[3] / num_par_n;
    extent_w = (inputs[0].adim[0] + num_par_w - 1) / num_par_w;
    extent_h = (inputs[0].adim[1] + num_par_h - 1) / num_par_h;
    extent_c = inputs[0].adim[2] / num_par_c;
    extent_n = inputs[0].adim[3] / num_par_n;
    assert(inputs[0].adim[2] % num_par_c == 0);
    assert(inputs[0].adim[3] % num_par_n == 0);
    //transform[0][0] = stride_w * output.pdim[0];
    //transform[1][1] = stride_h * output.pdim[1];
    //transform[2][2] = extent_nc;
    transform[0][0] = extent_w;
    transform[1][1] = extent_h;
    transform[2][2] = extent_c;
    transform[3][3] = extent_n;

    IndexPartition input_ip;
    {
      //int lo[4] = {0, 0, 0, 0};
      //int hi[4] = {extent_w-1, extent_h-1, extent_c-1, extent_n-1};
      Rect<4> extent_i(Realm::Point<4>(0, 0, 0, 0),
          Realm::Point<4>(extent_w-1, extent_h-1, extent_c-1, extent_n-1));
      input_ip = runtime->create_partition_by_restriction(ctx,
          input_is, task_is, transform, extent_i);
      assert(runtime->is_index_partition_disjoint(ctx, input_ip));
      assert(runtime->is_index_partition_complete(ctx, input_ip));
    }
    input_lps[0] = runtime->get_logical_partition(ctx, inputs[0].region, input_ip);
  }
#endif
}

Conv2D::Conv2D(FFModel& model,
               const std::string& pcname,
               int in_dim,
               int out_dim,
               int _kernel_h, int _kernel_w,
               int _stride_h, int _stride_w,
               int _padding_h, int _padding_w,
               ActiMode _activation,
               bool use_bias,
               Initializer* kernel_initializer,
               Initializer* bias_initializer)
: Op(pcname),
  in_channels(in_dim), out_channels(out_dim),
  kernel_h(_kernel_h), kernel_w(_kernel_w),
  stride_h(_stride_h), stride_w(_stride_w),
  padding_h(_padding_h), padding_w(_padding_w),
  activation(_activation), profiling(model.config.profiling)
{
  // Retrive the task indexspace for the op
  task_is = IndexSpaceT<4>(model.get_or_create_task_is(4, pcname));

  int input_c = in_dim;
  int output_c = out_dim;
  
  // Create kernel
  {
    const int dims[4] = {output_c, input_c, kernel_h, kernel_w};
    kernel = model.create_conv_weight<4>(dims, task_is, DT_FLOAT, kernel_initializer);
  }
  // Create bias tensor
  if (use_bias) {
    const int dims[1] = {output_c};
    bias = model.create_conv_weight<1>(dims, task_is, DT_FLOAT, bias_initializer);
  }
}

Tensor Conv2D::init_inout(FFModel& model, const Tensor& _input)
{
  add_to_model(model);
  assert(_input.numDim == 4);
  assert(_input.adim[2] == in_channels);
  inputs[0] = _input;
    // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<4>(model.get_or_create_task_is(4, pcname));

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // Create output tensor
  int input_w = _input.adim[0];
  int input_h = _input.adim[1];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_c = out_channels;
  int output_n = _input.adim[3];
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;
  {
    const int dims[4] = {output_n, output_c, output_h, output_w};
    output = model.create_tensor<4>(dims, task_is, DT_FLOAT);
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
        inputs[0], task_is, input_lps[0], input_grad_lps[0]);
  }
  return output;
}

void Conv2D::add_to_model(FFModel& model)
{
  model.layers.push_back(this);
  Parameter _kernel, _bias;
  _kernel.tensor = kernel;
  _kernel.op = this;
  model.parameters.push_back(_kernel);
  if (bias.numDim != 0) { // bias is used
    _bias.tensor = bias;
    _bias.op = this;
    model.parameters.push_back(_bias);
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
*/
__host__
OpMeta* Conv2D::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
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

  Conv2DMeta* m = new Conv2DMeta(handle);
  m->relu = conv->activation == AC_MODE_RELU;
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&m->biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&m->filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&m->convDesc));

  int input_w = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int input_h = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
  int output_w = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int output_h = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
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

  printf("filterDim: kernel(%d %d) c_in(%d), c_out(%d)\n", conv->kernel_h, conv->kernel_w, conv->inputs[0].pdim[2], conv->output.pdim[2]);
  checkCUDNN(cudnnSetFilter4dDescriptor(m->filterDesc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        conv->output.pdim[2],
                                        conv->inputs[0].pdim[2],
                                        conv->kernel_h,
                                        conv->kernel_w));

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
  m->fwdAlgo = selectConvolutionForwardAlgorithm(m->handle.dnn, m->inputTensor, acc_input.ptr,
                                                 m->filterDesc, acc_kernel.ptr, m->convDesc,
                                                 m->handle.workSpace, m->handle.workSpaceSize,
                                                 m->outputTensor, acc_output.ptr);
  // select backward filter algorithm
  m->bwdFilterAlgo = selectConvolutionBackwardFilterAlgorithm(
                         m->handle.dnn, m->inputTensor, acc_input.ptr,
                         m->outputTensor, acc_output.ptr,
                         m->convDesc, m->handle.workSpace, m->handle.workSpaceSize,
                         m->filterDesc, (void*)acc_kernel.ptr);
  // select backward data algorithm
  m->bwdDataAlgo = selectConvolutionBackwardDataAlgorithm(
                       m->handle.dnn, m->filterDesc, acc_kernel.ptr,
                       m->outputTensor, acc_output.ptr,
                       m->convDesc, m->handle.workSpace, m->handle.workSpaceSize,
                       m->inputTensor, (void*)acc_input.ptr);
  if (m->relu) {
    checkCUDNN(cudnnCreateActivationDescriptor(&m->actiDesc));
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
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(kernel.part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, kernel.region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(bias.part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, bias.region));
  launcher.add_field(3, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
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
  const Conv2D* conv = (Conv2D*) task->args;
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
  checkCUDNN(cudnnConvolutionForward(m->handle.dnn, &alpha,
                                     m->inputTensor, acc_input.ptr,
                                     m->filterDesc, acc_kernel.ptr,
                                     m->convDesc, m->fwdAlgo,
                                     m->handle.workSpace, m->handle.workSpaceSize,
                                     &beta, m->outputTensor, acc_output.ptr));

  checkCUDNN(cudnnAddTensor(m->handle.dnn, &alpha, m->biasTensor,
                            acc_bias.ptr, &alpha, m->outputTensor, acc_output.ptr));
  if (m->relu) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
                                      &alpha, m->outputTensor, acc_output.ptr,
                                      &beta, m->outputTensor, acc_output.ptr));
  }
  if (conv->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
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
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(kernel.part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, kernel.region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(bias.region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, bias.region));
  launcher.add_field(3, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): input_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): filter
  regions[5](O): filter_grad
  regions[6](O): bias_grad
*/
__host__
void Conv2D::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);
  float alpha = 1.0f, beta = 0.0f;
  const Conv2D* conv = (Conv2D*) task->args;
  const Conv2DMeta* m = *((Conv2DMeta**) task->local_args);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_input_grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, 4> acc_output(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_output_grad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime,
      true/*rreadOutput*/);
  TensorAccessorR<float, 4> acc_kernel(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_kernel_grad(
      regions[5], task->regions[5], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorW<float, 1> acc_bias_grad(
      regions[6], task->regions[6], FID_DATA, ctx, runtime,
      false/*readOutput*/);

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
  if (m->relu) {
    int n = acc_output.rect.volume();
    reluBackward<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(acc_output_grad.ptr, acc_output.ptr, n);
  }
  // Compute filter gradiant
  checkCUDNN(cudnnConvolutionBackwardFilter(m->handle.dnn, &alpha,
                                            m->inputTensor, acc_input.ptr,
                                            m->outputTensor, acc_output_grad.ptr,
                                            m->convDesc, m->bwdFilterAlgo,
                                            m->handle.workSpace, m->handle.workSpaceSize,
                                            &beta, m->filterDesc, acc_kernel_grad.ptr));
  // Compute bias gradiant
  checkCUDNN(cudnnConvolutionBackwardBias(m->handle.dnn, &alpha,
                                          m->outputTensor, acc_output_grad.ptr,
                                          &beta, m->biasTensor, acc_bias_grad.ptr));
  // no need to compute input_grad if we are the first layer
  if (!m->first_layer) {
    // Compute data gradiant
    checkCUDNN(cudnnConvolutionBackwardData(m->handle.dnn, &alpha,
                                            m->filterDesc, acc_kernel.ptr,
                                            m->outputTensor, acc_output_grad.ptr,
                                            m->convDesc, m->bwdDataAlgo,
                                            m->handle.workSpace, m->handle.workSpaceSize,
                                            &beta, m->inputTensor, acc_input_grad.ptr));
  }
  if (conv->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Conv2D backward time = %.2fms\n", elapsed);
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
  // regions[1](O): input_grad (we only need grad tensors)
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part_grad, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
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
      RegionRequirement(kernel.part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, kernel.region));
  launcher.add_field(4, FID_DATA);
  // regions[5](O): filter_grad
  launcher.add_region_requirement(
      RegionRequirement(kernel.part_grad, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, kernel.region_grad));
  launcher.add_field(5, FID_DATA);
  // regions[6](O): bias_grad
  launcher.add_region_requirement(
      RegionRequirement(bias.part_grad, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, bias.region_grad));
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

__host__
Tensor* Conv2D::get_weight()
{
  return &kernel;
}

__host__
Tensor* Conv2D::get_bias()
{
  return &bias;
}

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
  RegionRequirement kernel_req(kernel.region, READ_WRITE, EXCLUSIVE, kernel.region);
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
  RegionRequirement bias_req(bias.region, READ_WRITE, EXCLUSIVE, bias.region);
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

__host__
void Conv2D::print_layer_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);

  TensorAccessorR<float, 4> acc_kernel(regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 1> acc_bias(regions[1], task->regions[1], FID_DATA, ctx, runtime);
  
  const float *kernel_ptr = acc_kernel.ptr;
  const float *bias_ptr = acc_bias.ptr;
  
  size_t kernel_size = acc_kernel.rect.volume();
  size_t bias_size = acc_bias.rect.volume();
  printf("kernel, %d, %p\n", kernel_size, kernel_ptr);
  printf("bias, %d, %p\n", bias_size, bias_ptr);
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
