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

Tensor CnnModel::add_conv_layer(Tensor input, int out_channels,
                                int kernel_x, int kernel_y,
                                int stride_x, int stride_y,
                                int padding_x, int padding_y)
{
  assert(input.numDim == 4); /*NCHW*/
  int in_channels = input.dim[2];
  Conv2D *conv = new Conv2D(config, input, in_channels, out_channels, kernel_x, kernel_y,
                            stride_x, stride_y, padding_x, padding_y);
  layers.push_back(conv);
  return conv->output;
}

/*
locals[0] = kernel
locals[1] = bias
*/
Conv2D::Conv2D(CnnConfig config, Tensor input,
               int _in_channels, int _out_channels,
               int kernel_h, int kernel_w,
               int stride_h, int stride_w,
               int padding_h, int padding_w)
: Op(input), in_channels(_in_channels), out_channels(_out_channels)
{
  Context ctx = config.lg_ctx;
  HighLevelRuntime* runtime = config.lg_hlr;
  kernel[0] = kernel_w;
  kernel[1] = kernel_h;
  stride[0] = stride_w;
  stride[1] = stride_h;
  padding[0] = padding_w;
  padding[1] = padding_h;
  // Create output tensor
  int input_w = input.dim[0];
  int input_h = input.dim[1];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_nc = input.dim[3] * out_channels;
  FieldSpace fs;
  fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }
  Realm::ZRect<3, coord_t> color_bounds(Realm::ZPoint<3>(0, 0, 0),
                       Realm::ZPoint<3>(config.num_par_w-1, config.num_par_h-1, config.num_par_n-1));
  IndexSpaceT<3> color_is = runtime->create_index_space(ctx, color_bounds);

  Realm::ZRect<3, coord_t> output_rect(Realm::ZPoint<3>(0, 0, 0),
                      Realm::ZPoint<3>(output_w-1, output_h-1, output_nc-1));
  IndexSpaceT<3> output_is = runtime->create_index_space(ctx, output_rect);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  Realm::ZMatrix<3, 3, coord_t> transform;
  int extent_w = (output_w + config.num_par_w - 1) / config.num_par_w;
  int extent_h = (output_h + config.num_par_h - 1) / config.num_par_h;
  int extent_nc = output_nc / config.num_par_n;
  assert(output_nc % config.num_par_n == 0);
  Realm::ZRect<3, coord_t> extent(Realm::ZPoint<3>(0, 0, 0),
                    Realm::ZPoint<3>(extent_w, extent_h, extent_nc));
  transform[0][0] = extent_w;
  transform[1][1] = extent_h;
  transform[2][2] = extent_nc;
  IndexPartition output_ip =
    runtime->create_partition_by_restriction(ctx, output_is, color_is, transform, extent);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);

  int kernel_nc = config.num_workers * in_channels * out_channels;
  Realm::ZRect<1, coord_t> kernel_rect(0, kernel_w * kernel_h * kernel_nc - 1);
  IndexSpaceT<1> kernel_is = runtime->create_index_space(ctx, kernel_rect);
  LogicalRegion kernel_lr = runtime->create_logical_region(ctx, kernel_is, fs);
  IndexPartition kernel_ip = runtime->create_equal_partition(ctx, kernel_is, color_is);
  LogicalPartition kernel_lp = runtime->get_logical_partition(ctx, kernel_lr, kernel_ip);
  TensorWithGrad kernel_tensor;
  kernel_tensor.region = kernel_lr;
  kernel_tensor.partition = kernel_lp;
  locals[0] = kernel_tensor;

  int bias_nc = config.num_workers * out_channels;
  Realm::ZRect<1, coord_t> bias_rect(0, bias_nc - 1);
  IndexSpaceT<1> bias_is = runtime->create_index_space(ctx, bias_rect);
  LogicalRegion bias_lr = runtime->create_logical_region(ctx, bias_is, fs);
  IndexPartition bias_ip = runtime->create_equal_partition(ctx, bias_is, color_is);
  LogicalPartition bias_lp = runtime->get_logical_partition(ctx, bias_lr, bias_ip);
  TensorWithGrad bias_tensor;
  bias_tensor.region = bias_lr;
  bias_tensor.partition = bias_lp;
  locals[1] = bias_tensor;

  inputs[0] = input;
  output.numDim = 4;
  output.dim[0] = output_w;
  output.dim[1] = output_h;
  output.dim[2] = out_channels;
  output.dim[3] = input.dim[3];
  output.region = output_lr;
  output.partition = output_lp;
}

/*
  regions[0]: output
  regions[1]: filter
  regions[2]: bias
*/
OpMeta* Conv2D::init(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Conv2D* conv = (Conv2D*) task->local_args;
  Conv2DMeta* m = new Conv2DMeta();
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&m->biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&m->filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&m->convDesc));

  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        conv->inputs[0].dim[0],
                                        conv->inputs[0].dim[1],
                                        conv->inputs[0].dim[2],
                                        conv->inputs[0].dim[3]));
  
  checkCUDNN(cudnnSetTensor4dDescriptor(m->biasTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        1,
                                        conv->output.dim[1],
                                        1,
                                        1));

  checkCUDNN(cudnnSetFilter4dDescriptor(m->filterDesc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        conv->output.dim[1],
                                        conv->inputs[0].dim[1],
                                        conv->kernel[0],
                                        conv->kernel[1]));

  checkCUDNN(cudnnSetConvolution2dDescriptor(m->convDesc,
                                             conv->padding[0],
                                             conv->padding[1],
                                             conv->stride[0],
                                             conv->stride[1],
                                             1/*upscale_x*/,
                                             1/*upscale_y*/,
                                             CUDNN_CROSS_CORRELATION));

  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(m->convDesc,
                                                   m->inputTensor,
                                                   m->filterDesc,
                                                   &n, &c, &h, &w));
  assert(n == conv->output.dim[0]);
  assert(c == conv->output.dim[1]);
  assert(h == conv->output.dim[2]);
  assert(w == conv->output.dim[3]);

  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  return m;
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I): filter
  regions[3](I): bias
*/
__host__
void Conv2D::forward(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f, beta = 0.0f;
  const Conv2DMeta* m = (Conv2DMeta*) task->local_args;
  const FieldAccessor<READ_ONLY, float, 3> acc_input(regions[0], FID_DATA);
  const FieldAccessor<WRITE_DISCARD, float, 3> acc_output(regions[1], FID_DATA);
  const FieldAccessor<READ_ONLY, float, 3> acc_filter(regions[2], FID_DATA);
  const FieldAccessor<READ_ONLY, float, 3> acc_bias(regions[3], FID_DATA);
  Realm::ZRect<3> rect[4];
  for (int i = 0; i < 4; i++)
    rect[i] = runtime->get_index_space_domain(ctx,
      task->regions[i].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect[0]));
  assert(acc_output.accessor.is_dense_arbitrary(rect[1]));
  assert(acc_filter.accessor.is_dense_arbitrary(rect[2]));
  assert(acc_bias.accessor.is_dense_arbitrary(rect[3]));
  const float *input_ptr = acc_input.ptr(rect[0].lo);
  float *output_ptr = acc_output.ptr(rect[1].lo);
  const float *filter_ptr = acc_filter.ptr(rect[2].lo);
  const float *bias_ptr = acc_bias.ptr(rect[3].lo);
  
  checkCUDNN(cudnnConvolutionForward(m->handle.dnn, &alpha,
                                     m->inputTensor, input_ptr,
                                     m->filterDesc, filter_ptr,
                                     m->convDesc, m->fwdAlgo,
                                     m->handle.workSpace, m->handle.workSpaceSize,
                                     &beta, m->outputTensor, output_ptr));

  checkCUDNN(cudnnAddTensor(m->handle.dnn, &alpha, m->biasTensor,
                            bias_ptr, &alpha, m->outputTensor, output_ptr));
}

/*
  regions[0](I/O): input
  regions[1](I): output
  regions[2](I/O): filter
  regions[3](I/O): bias
*/
__host__
void Conv2D::backward(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f, beta = 0.0f;
}

