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

Conv2D::Conv2D(int _in_channels, int _out_channels, int kernel_x, int kernel_y,
               int stride_x, int stride_y, int padding_x, int padding_y,
               Op* pre_op)
: Op(pre_op), in_channels(_in_channels), out_channels(_out_channels)
{
  kernel[0] = kernel_x;
  kernel[1] = kernel_y;
  stride[0] = stride_x;
  stride[1] = stride_y;
  padding[0] = padding_x;
  padding[1] = padding_y;
}

/*
  regions[0]: output
  regions[1]: filter
  regions[2]: bias
*/
__host__
OpMeta Conv2D::init(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
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
                                        conv->inputs[0].dim[3]);
  
  checkCUDNN(cudnnSetTensor4dDescriptor(m->biasTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        1,
                                        conv->output.dim[1],
                                        1,
                                        1);

  checkCUDNN(cudnnSetFilter4dDescriptor(m->filterDesc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        conv->output.dim[1],
                                        conv->inputs[0].dim[1],
                                        conv->kernel[0],
                                        conv->kernel[1]);

  checkCUDNN(cudnnSetConvolution2dDescriptor(m->convDesc,
                                             conv->padding[0],
                                             conv->padding[1],
                                             conv->stride[0],
                                             conv->stride[1],
                                             1/*upscale_x*/,
                                             1/*upscale_y*/,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT);

  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(m->convDesc,
                                                   conv->inputTensor,
                                                   conv->filterDesc,
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
                     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f, beta = 0.0f;
  const Conv2D* conv = (Conv2D*) task->local_args;
  const Conv2DMeta* m = (Conv2DMeta*) conv->opsMeta;
  const FieldAccessor<READ_ONLY, float, 3> acc_input(regions[0], FID_DATA);
  const FieldAccessor<WRITE_DISCARD, float, 3> acc_output(regions[1], FID_DATA);
  const FieldAccessor<READ_ONLY, float, 3> acc_filter(regions[2], FID_DATA);
  const FieldAccessor<READ_ONLY, float, 3> acc_bias(regions[3], FID_DATA);
  Rect<3> rect[4];
  for (int i = 0; i < 4; i++)
    rect[i] = runtime->get_index_space_domain(ctx,
      task->regions[i].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect[0]));
  assert(acc_output.accessor.is_dense_arbitrary(rect[1]));
  assert(acc_filter.accessor.is_dense_arbitrary(rect[2]));
  assert(acc_bias.accessor.is_dense_arbitrary(rect[3]));
  const float *input_ptr = acc_input.ptr(rect[0].lo);
  float *output_ptr = acc_input.ptr(rect[0].lo);
  const float *filter_ptr = acc_input.ptr(rect[0].lo);
  const float *bias_ptr = acc_input.ptr(rect[0].lo);
  
  chcekCUDNN(cudnnConvolutionForward(conv->dnnHandle, &alpha, m->inputTensor,
                                     
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
                      Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
}
