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

#include "ops/conv_2d.h"
#include "cuda_helper.h"
#include "hash_utils.h"

using namespace Legion;
Tensor FFModel::conv2d(const Tensor input,
                       int outChannels,
                       int kernelH, int kernelW,
                       int strideH, int strideW,
                       int paddingH, int paddingW,
                       ActiMode activation,
                       int groups,
                       bool use_bias,
                       const Op* shared_op,
                       Initializer* kernel_initializer,
                       Initializer* bias_initializer,
                       char const *name)
{
  assert(input->num_dims == 5); /*RNCHW*/

  Conv2D *conv = new Conv2D(
      *this, 
      input, 
      outChannels,
      kernelH, kernelW,
      strideH, strideW, 
      paddingH, paddingW, 
      activation,
      groups,
      use_bias,
      false,
      name
  );
  layers.push_back(conv);
  return conv->outputs[0];
}

namespace Input {
  static constexpr int INDEX = 0;

  enum {
    WIDTH = 0,
    HEIGHT = 1,
    CHANNEL = 2,
    SAMPLE = 3,
    REPLICA = 4,
    NUMDIM
  };
}

namespace Output {
  enum {
    WIDTH = 0,
    HEIGHT = 1,
    CHANNEL = 2,
    SAMPLE = 3,
    REPLICA = 4,
    NUMDIM
  };
}

namespace Kernel {
  static constexpr int INDEX = 0;

  enum {
    WIDTH = 0,
    HEIGHT = 1,
    CHANNEL_IN = 2,
    CHANNEL_OUT = 3,
    REPLICA = 4,
    NUMDIM
  };
}

namespace Bias {
  static constexpr int INDEX = 1;

  enum {
    CHANNEL = 0,
    REPLICA_1 = 1,
    REPLICA_2 = 2,
    REPLICA_3 = 3,
    REPLICA_4 = 4,
    NUMDIM
  };
}


Conv2DParams Conv2D::get_params() const {
  Conv2DParams params;
  params.out_channels = this->out_channels;
  params.kernel_h = this->kernel_h;
  params.kernel_w = this->kernel_w;
  params.stride_h = this->stride_h;
  params.stride_w = this->stride_w;
  params.padding_h = this->padding_h;
  params.padding_w = this->padding_w;
  params.activation = this->activation;
  params.groups = this->groups;
  params.use_bias = this->use_bias;

  return params;
}

Node FFModel::get_or_create_conv2d_node(const Tensor input,
                                        const Conv2DParams& params) 
{
  if (!params.is_valid(input)) {
    return Node::INVALID_NODE;
  }

  size_t hash = input->get_owner_independent_hash();
  hash_combine(hash, params.out_channels);
  hash_combine(hash, params.kernel_h);
  hash_combine(hash, params.kernel_w);
  hash_combine(hash, params.stride_h);
  hash_combine(hash, params.stride_w);
  hash_combine(hash, params.padding_h);
  hash_combine(hash, params.padding_w);
  hash_combine(hash, params.activation);
  hash_combine(hash, params.groups);
  hash_combine(hash, params.use_bias);

  Conv2D *conv = NULL;

  const auto &it = this->cached_conv2d_ops.find(hash);
  if (it != cached_conv2d_ops.end()) {
    conv = it->second;
  } else {
    conv = new Conv2D(*this, 
                      input, 
                      params.out_channels, 
                      params.kernel_h, params.kernel_w, 
                      params.stride_h, params.stride_w,
                      params.padding_h, params.padding_w,
                      params.activation, 
                      params.groups, 
                      params.use_bias,
                      false/*allocate_weights*/,
                      NULL);
    cached_conv2d_ops[hash] = conv;
  }

  return this->new_node(conv);
}

Node FFModel::get_or_create_conv2d_node(const Tensor input,
                                        int outChannels,
                                        int kernelH, int kernelW,
                                        int strideH, int strideW,
                                        int paddingH, int paddingW,
                                        ActiMode activation,
                                        int groups,
                                        bool use_bias) 
{
  Conv2DParams params;
  params.out_channels = outChannels;
  params.kernel_h = kernelH;
  params.kernel_w = kernelW;
  params.stride_h = strideH;
  params.stride_w = strideW;
  params.padding_h = paddingH;
  params.padding_w = paddingW;
  params.activation = activation;
  params.groups = groups;
  params.use_bias = use_bias;

  return this->get_or_create_conv2d_node(input, params);
}

void Conv2DParams::mark_replica_dims(const Tensor input,
                               ParallelDim output_dims[MAX_TENSOR_DIM], 
                               ParallelDim kernel_dims[MAX_TENSOR_DIM], 
                               ParallelDim bias_dims[MAX_TENSOR_DIM]) const 
{
  if (output_dims != nullptr) {
    output_dims[Output::REPLICA].is_replica_dim = true;
  }
  if (kernel_dims != nullptr) {
    kernel_dims[Output::REPLICA].is_replica_dim = true;
  }
  if (bias_dims != nullptr) {
    bias_dims[Bias::REPLICA_1].is_replica_dim = true;
    bias_dims[Bias::REPLICA_2].is_replica_dim = true;
    bias_dims[Bias::REPLICA_3].is_replica_dim = true;
    bias_dims[Bias::REPLICA_4].is_replica_dim = true;
  }
}

int Conv2DParams::output_size(const Tensor input, ParallelDim output_dims[MAX_TENSOR_DIM]) const {
  int input_w = input->dims[Input::WIDTH].size;
  int input_h = input->dims[Input::HEIGHT].size;

  output_dims[Output::SAMPLE].size = input->dims[Input::SAMPLE].size;
  output_dims[Output::CHANNEL].size = out_channels;
  output_dims[Output::HEIGHT].size = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  output_dims[Output::WIDTH].size = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;

  return input->num_dims;
};

int Conv2DParams::kernel_size(const Tensor input, ParallelDim kernel_dims[MAX_TENSOR_DIM]) const {
  kernel_dims[Kernel::CHANNEL_OUT].size = this->out_channels;
  kernel_dims[Kernel::CHANNEL_IN].size = input->dims[Input::CHANNEL].size / this->groups;
  kernel_dims[Kernel::HEIGHT].size = this->kernel_h * input->dims[Input::HEIGHT].degree;
  kernel_dims[Kernel::WIDTH].size = this->kernel_w * input->dims[Input::WIDTH].degree;

  return Kernel::NUMDIM;
}

int Conv2DParams::bias_size(const Tensor input, ParallelDim bias_dims[MAX_TENSOR_DIM]) const {
  bias_dims[Bias::CHANNEL].size = this->out_channels;

  return Bias::NUMDIM;
};

void Conv2DParams::solve_dims(const Tensor input, 
                              ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                              ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,  
                              ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const 
{
  assert ((output_dims == nullptr) == (output_ndims == nullptr));
  assert ((kernel_dims == nullptr) == (kernel_ndims == nullptr));
  assert ((bias_dims == nullptr) == (bias_ndims == nullptr));

  std::vector<ParallelDimMappingRecord> mapping;
  Conv2D::construct_mappings(mapping, this->use_bias);

  this->mark_replica_dims(input, output_dims, kernel_dims, bias_dims);

  std::vector<ParallelDim *> output_dim_sets;
  if (output_dims != nullptr) {
    output_dim_sets.push_back(output_dims);
  }

  std::vector<ParallelDim *> weight_dim_sets;
  if (kernel_dims != nullptr) {
    weight_dim_sets.push_back(kernel_dims);
  }
  if (bias_dims != nullptr && this->use_bias) {
    weight_dim_sets.push_back(bias_dims);
  }

  solve_parallel_dim_mappings(
      mapping, 
      {input->dims},
      weight_dim_sets,
      output_dim_sets
  );

  if (output_dims != nullptr) {
    *output_ndims = this->output_size(input, output_dims);
  }
  if (kernel_dims != nullptr) {
    *kernel_ndims = this->kernel_size(input, kernel_dims);
  }
  if (bias_dims != nullptr && this->use_bias) {
    *bias_ndims = this->bias_size(input, bias_dims);
  }
}

/*static*/
void Conv2D::construct_mappings(std::vector<ParallelDimMappingRecord>& out, bool use_bias) {
  Conv2D::construct_output_mappings(out);
  Conv2D::construct_weight_mappings(out, use_bias);
}

/*static*/
void Conv2D::construct_output_mappings(std::vector<ParallelDimMappingRecord>& out) {
  Op::construct_output_parallel_dims(
    out, 
    {
      {Input::CHANNEL, MappingOperation::REPLICATE, Output::REPLICA},
      {Input::SAMPLE, MappingOperation::PARTITION, Output::SAMPLE},
      {Input::REPLICA, MappingOperation::PARTITION, Output::CHANNEL},
      {Input::HEIGHT, MappingOperation::PARTITION, Output::HEIGHT},
      {Input::WIDTH, MappingOperation::PARTITION, Output::WIDTH}
    }
  );
}

/*static*/
void Conv2D::construct_weight_mappings(std::vector<ParallelDimMappingRecord>& out, bool use_bias) {
  Op::construct_weight_parallel_dims(
    out,
    {
      {Input::REPLICA, MappingOperation::PARTITION, Kernel::CHANNEL_OUT},
      {Input::SAMPLE, MappingOperation::REPLICATE, Kernel::REPLICA},
      {Input::CHANNEL, MappingOperation::PARTITION, Kernel::CHANNEL_IN}, 
      {Input::HEIGHT, MappingOperation::REPLICATE, Kernel::HEIGHT}, // Kernel::{HEIGHT, WEIGHT} would both work here
      {Input::WIDTH, MappingOperation::REPLICATE, Kernel::WIDTH}, // same as above
    }, 
    Input::INDEX, Kernel::INDEX
  );

  if (use_bias) {
    Op::construct_weight_parallel_dims(
      out,
      {
        {Input::REPLICA, Bias::REPLICA_1},
        {Input::SAMPLE, Bias::REPLICA_2},
        {Input::CHANNEL, Bias::CHANNEL},
        {Input::HEIGHT, Bias::REPLICA_3},
        {Input::WIDTH, Bias::REPLICA_4}
      }, 
      Input::INDEX, Bias::INDEX
    );
  }
}

Conv2D::Conv2D(FFModel& model,
               Conv2D const &other,
               const Tensor input,
               bool allocate_weights)
: Conv2D(model, 
         input, 
         other.out_channels, 
         other.kernel_h,
         other.kernel_w,
         other.stride_h,
         other.stride_w,
         other.padding_h,
         other.padding_w,
         other.activation,
         other.groups,
         allocate_weights,
         other.use_bias,
         other.name) 
{ }

bool Conv2DParams::is_valid(const Tensor input) const {
  ParallelDim output_dims[MAX_TENSOR_DIM],
              kernel_dims[MAX_TENSOR_DIM],
              bias_dims[MAX_TENSOR_DIM];
  int output_ndims, 
      kernel_ndims,
      bias_ndims;

  this->solve_dims(
      input, 
      output_dims, &output_ndims, 
      kernel_dims, &kernel_ndims,
      bias_dims, &bias_ndims
  );

  bool is_valid = true;
  is_valid &= input->check_valid();
  is_valid &= ParallelDim::dims_are_valid(output_dims, output_ndims);
  is_valid &= ParallelDim::dims_are_valid(kernel_dims, kernel_ndims);
  if (use_bias) { 
    is_valid &= ParallelDim::dims_are_valid(bias_dims, bias_ndims);
  }

  return is_valid;
}

Conv2D::Conv2D(FFModel& model,
               const Tensor input,
               int outChannels,
               int kernelH, int kernelW,
               int strideH, int strideW, 
               int paddingH, int paddingW,
               ActiMode activation,
               int groups,
               bool allocate_weights,
               bool use_bias,
               const char* name)
: Op(model, OP_CONV2D, name, 1/*inputs*/, use_bias ? 2 : 1/*weights*/, allocate_weights, 1/*outputs*/, input),
  in_channels(input->dims[Input::CHANNEL].size),
  out_channels(outChannels),
  kernel_h(kernelH), kernel_w(kernelW),
  stride_h(strideH), stride_w(strideW),
  padding_h(paddingH), padding_w(paddingW),
  activation(activation),
  groups(groups),
  use_bias(use_bias)
{
  assert (input->num_dims == Input::NUMDIM);
  assert (this->stride_h > 0);
  assert (this->stride_w > 0);

  ParallelDim output_dims[MAX_TENSOR_DIM],
              kernel_dims[MAX_TENSOR_DIM], 
              bias_dims[MAX_TENSOR_DIM];
  int output_ndims,
      kernel_ndims,
      bias_ndims;

  this->construct_mappings(
      *this->parallel_dims_mapping, this->use_bias);
  this->get_params().solve_dims(
      this->inputs[0],
      output_dims, &output_ndims,
      kernel_dims, &kernel_ndims,
      bias_dims, &bias_ndims);

  if (allocate_weights) {
    Initializer *kernel_initializer = new GlorotUniform(std::rand()/*seed*/);

    weights[Kernel::INDEX] = model.create_weight_legion_ordering(
        kernel_ndims, kernel_dims, DT_FLOAT, NULL/*owner_op*/, true/*create_grad*/, kernel_initializer, CHOSEN_SYNC_TYPE);
    
    if (use_bias) {
      Initializer *bias_initializer = new ZeroInitializer();

      weights[Bias::INDEX] = model.create_weight_legion_ordering(
          bias_ndims, bias_dims, DT_FLOAT, NULL/*owner_op*/, true/*create_grad*/, bias_initializer, CHOSEN_SYNC_TYPE);
    }
  }

  outputs[0] = model.create_tensor_legion_ordering(output_ndims, output_dims, DT_FLOAT, this);

  assert(check_output_input_weight_parallel_dims(allocate_weights));
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
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const Conv2D* conv = (Conv2D*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  TensorAccessorR<float, Input::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Output::NUMDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, Kernel::NUMDIM> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  // TensorAccessorR<float, 1> acc_bias(
  //     regions[3], task->regions[3], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Kernel::NUMDIM> acc_kernel_grad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  //TensorAccessorW<float, 4> acc_input_grad(
  //    regions[4], task->regions[4], FID_DATA, ctx, runtime,
  //    false/*readOutput*/);

  Conv2DMeta* m = new Conv2DMeta(handle);
  m->relu = conv->activation == AC_MODE_RELU;
  m->use_bias = conv->use_bias;
  m->profiling = conv->profiling;
  std::strcpy(m->op_name, conv->name);

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

  // Require that input_c is divisible by conv->groups
  assert(input_c % conv->groups == 0);
  printf("filterDim: kernel(%d %d) c_in(%d), c_out(%d)\n",
      conv->kernel_h, conv->kernel_w, input_c / conv->groups, output_c);
  checkCUDNN(cudnnSetFilter4dDescriptor(m->filterDesc,
      CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
      output_c, input_c / conv->groups, conv->kernel_h, conv->kernel_w));

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
  if (conv->groups != 1) {
    checkCUDNN(cudnnSetConvolutionGroupCount(m->convDesc, conv->groups));
  }

  // enable tensor core when possible
  if (m->handle.allowTensorOpMathConversion) {
    checkCUDNN(cudnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    checkCUDNN(cudnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH));
  }

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
                       m->inputTensor, (float*)acc_input.ptr);
  if (m->relu) {
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
  }
  return m;
}

void Conv2D::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(CONV2D_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Conv2D)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(2, FID_DATA);
  // launcher.add_region_requirement(
  //     RegionRequirement(weights[1]->part, 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, weights[1]->region));
  // launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part_grad, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, weights[0]->region_grad));
  launcher.add_field(3, FID_DATA);
  //launcher.add_region_requirement(
  //    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
  //                      WRITE_ONLY, EXCLUSIVE, inputs[0]->region_grad));
  //launcher.add_field(4, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

/*static*/
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

  // use_bias == True
  if (bias_ptr != NULL) {
    checkCUDNN(cudnnAddTensor(m->handle.dnn, &alpha, m->biasTensor,
                              bias_ptr, &alpha, m->outputTensor, output_ptr));
  }
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
  //Conv2D* conv = (Conv2D*) task->args;
  const Conv2DMeta* m = *((Conv2DMeta**) task->local_args);
  assert(regions.size() == (3 + int(m->use_bias)));
  assert(task->regions.size() == (3 + int(m->use_bias)));
  TensorAccessorR<float, Input::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Output::NUMDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, Kernel::NUMDIM> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  const float* acc_bias_ptr = NULL;
  if (m->use_bias) { 
    TensorAccessorR<float, Bias::NUMDIM> acc_bias(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
    acc_bias_ptr = acc_bias.ptr;
  }

  //printf("fwdAlgo(%d), bwdFilterALgo(%d), bwdDataAlgo(%d)\n", (int)m->fwdAlgo,(int) m->bwdFilterAlgo,(int) m->bwdDataAlgo);
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
  Conv2D::forward_kernel(m, acc_input.ptr, acc_output.ptr, acc_kernel.ptr, acc_bias_ptr);
  if (m->profiling) {
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
    printf("%s [Conv2D] forward time (CF) = %.2fms\n", m->op_name, elapsed);
  }
}

__host__
void Conv2D::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(CONV2D_FWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(2, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(
        RegionRequirement(weights[1]->region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, weights[1]->region));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*static*/
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
  if (bias_grad_ptr != NULL) {
    checkCUDNN(cudnnConvolutionBackwardBias(m->handle.dnn, &alpha,
                                            m->outputTensor, output_grad_ptr,
                                            &alpha, m->biasTensor, bias_grad_ptr));
                                          }
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
  //Conv2D* conv = (Conv2D*) task->args;
  const Conv2DMeta* m = *((Conv2DMeta**) task->local_args);
  assert(regions.size() == (6 + int(m->use_bias)));
  assert(task->regions.size() == (6 + int(m->use_bias)));
  TensorAccessorR<float, Input::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Input::NUMDIM> acc_input_grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, Output::NUMDIM> acc_output(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Output::NUMDIM> acc_output_grad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, Kernel::NUMDIM> acc_kernel(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Kernel::NUMDIM> acc_kernel_grad(
      regions[5], task->regions[5], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  float* acc_bias_grad_ptr = NULL;
  if (m->use_bias) { 
    TensorAccessorW<float, Bias::NUMDIM> acc_bias_grad(
        regions[6], task->regions[6], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    acc_bias_grad_ptr = static_cast<float*>(acc_bias_grad.ptr);
  }
  

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
  Conv2D::backward_kernel(m, acc_input.ptr, acc_input_grad.ptr,
                          acc_output.ptr, acc_output_grad.ptr,
                          acc_kernel.ptr, acc_kernel_grad.ptr,
                          acc_bias_grad_ptr);
  if (m->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [Conv2D] backward time = %.2fms\n", m->op_name, elapsed);
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
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(CONV2D_BWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input_grad
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
      RegionRequirement(weights[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(4, FID_DATA);
  // regions[5](I/O): filter_grad
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, weights[0]->region_grad));
  launcher.add_field(5, FID_DATA);
  if (use_bias) {
    // regions[6](I/O): bias_grad
    launcher.add_region_requirement(
        RegionRequirement(weights[1]->part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, weights[1]->region_grad));
    launcher.add_field(6, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  // TODO: remove this line
  //if (first_layer)
    //fm.wait_all_results();
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
    RegionRequirement(kernel->region, READ_ONLY, EXCLUSIVE, kernel->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(bias->region, READ_ONLY, EXCLUSIVE, bias->region));
  launcher.add_field(1, FID_DATA);
  Future fu = runtime->execute_task(ctx, launcher);
  fu.wait();
#else
  RegionRequirement kernel_req(weights[0]->region, READ_WRITE, EXCLUSIVE, weights[0]->region);
  kernel_req.add_field(FID_DATA);
  InlineLauncher kernel_launcher(kernel_req);
  PhysicalRegion kernel_region = runtime->map_region(ctx, kernel_launcher);
  kernel_region.wait_until_valid();

/*
  RegionRequirement kernel_grad_req(kernel->region_grad, READ_WRITE, EXCLUSIVE, kernel->region_grad);
  kernel_grad_req.add_field(FID_DATA);
  InlineLauncher kernel_grad_launcher(kernel_grad_req);
  PhysicalRegion kernel_grad_region = runtime->map_region(ctx, kernel_grad_launcher);
  kernel_grad_region.wait_until_valid();
*/
  RegionRequirement bias_req(weights[1]->region, READ_WRITE, EXCLUSIVE, weights[1]->region);
  bias_req.add_field(FID_DATA);
  InlineLauncher bias_launcher(bias_req);
  PhysicalRegion bias_region = runtime->map_region(ctx, bias_launcher);
  bias_region.wait_until_valid();
/*
  RegionRequirement bias_grad_req(bias->region_grad, READ_WRITE, EXCLUSIVE, bias->region_grad);
  bias_grad_req.add_field(FID_DATA);
  InlineLauncher bias_grad_launcher(bias_grad_req);
  PhysicalRegion bias_grad_region = runtime->map_region(ctx, bias_grad_launcher);
  bias_grad_region.wait_until_valid();
  */
  TensorAccessorW<float, Kernel::NUMDIM> acc_kernel(kernel_region, kernel_req, FID_DATA, ctx, runtime, true);
//  const AccessorRW<float, 1> acc_kernel_grad(kernel_grad_region, FID_DATA);
  TensorAccessorW<float, Bias::NUMDIM> acc_bias(bias_region, bias_req, FID_DATA, ctx, runtime, true);
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
  printf("kernel, %p, %zu, [%d, %d, %d, %d]\n", kernel_ptr, kernel_size, kernel_dim1, kernel_dim2, kernel_dim3, kernel_dim4);
  //printf("kernel_grad, %d\n", kernel_grad_size);
  printf("bias, %p, %zu\n", bias_ptr, bias_size);
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

bool Conv2D::measure_operator_cost(Simulator* sim,
                                   const ParallelConfig& pc,
                                   CostMetrics& cost_metrics) const
{
  TensorBase sub_output, sub_input;
  if(!outputs[0]->get_output_sub_tensor(pc, sub_output, OP_CONV2D))
    return false;
  if(!inputs[0]->get_input_sub_tensor(pc, sub_input, OP_CONV2D))
    return false;
  int input_w = sub_input.dims[0].size;
  int input_h = sub_input.dims[1].size;
  int input_c = sub_input.dims[2].size;
  int input_n = sub_input.dims[3].size;
  int output_w = sub_output.dims[0].size;
  int output_h = sub_output.dims[1].size;
  int output_c = sub_output.dims[2].size;
  int output_n = sub_output.dims[3].size;
  int pad_h = ((output_h - 1) * stride_h + kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * stride_w + kernel_w - input_w + 1) / 2;

  Conv2DMeta* m = sim->conv2d_meta;
  m->relu = activation == AC_MODE_RELU;
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));
  checkCUDNN(cudnnSetTensor4dDescriptor(m->biasTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, 1, output_c, 1, 1));
  // require input_c is divisible by groups
  assert(input_c % groups == 0);
  checkCUDNN(cudnnSetFilter4dDescriptor(m->filterDesc, CUDNN_DATA_FLOAT,
      CUDNN_TENSOR_NCHW, output_c, input_c / groups, kernel_h, kernel_w));
  checkCUDNN(cudnnSetConvolution2dDescriptor(m->convDesc, pad_h, pad_w,
      stride_h, stride_w, 1/*dilationH*/, 1/*dilationW*/,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  checkCUDNN(cudnnSetConvolutionGroupCount(m->convDesc, groups));
  if (m->handle.allowTensorOpMathConversion) {
    checkCUDNN(cudnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    checkCUDNN(cudnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH));
  }
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
  float* weight_ptr = (float*)sim->allocate((size_t)output_c * input_c * kernel_h * kernel_w / groups, DT_FLOAT);
  assert(weight_ptr != NULL);
  float* bias_ptr = (float*)sim->allocate(output_c, DT_FLOAT);
  assert(bias_ptr != NULL);

  // compute memory usage
  // Assume:
  //   1. all memory allocations use Simulator::allocate
  //   2. we call Simulator::free_all before measure an operator
  // Therefore, the memory usage of an operator is sim->offset
  cost_metrics.memory_requirement = (size_t)sim->offset;

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
    cost_metrics.forward_time = perfResults[0].time;
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
    cost_metrics.backward_time = perfResults[0].time;
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
    cost_metrics.backward_time += perfResults[0].time;
  }
  printf("[Measure Conv2D] name(%s) input(%d %d %d %d) weight(%d %d %d %d) output(%d %d %d %d) stride(%d %d) padding(%d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
         name,
         input_n, input_c, input_h, input_w,
         output_c, input_c / groups, kernel_h, kernel_w,
         output_n, output_c, output_h, output_w,
         stride_h, stride_w,
         padding_h, padding_w,
         cost_metrics.forward_time, cost_metrics.backward_time);
  return true;
}

bool Conv2D::estimate_sync_cost(Simulator* sim, 
                                const MachineView& view,
                                CostMetrics& cost_metrics) const 
{
  ParallelDim kernel_dims[MAX_TENSOR_DIM],
              bias_dims[MAX_TENSOR_DIM];
  int kernel_ndims,
      bias_ndims;
  
  this->get_params().solve_dims(this->inputs[0], 
                                nullptr, nullptr,
                                kernel_dims, &kernel_ndims,
                                bias_dims, &bias_ndims);

  cost_metrics.sync_time = sim->default_estimate_sync_cost(kernel_dims, kernel_ndims, view);

  if (this->use_bias) {
    cost_metrics.sync_time += sim->default_estimate_sync_cost(bias_dims, bias_ndims, view);
  }

  return true;
}
