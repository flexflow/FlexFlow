/* Copyright 2022 NVIDIA CORPORATION
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

#ifndef __LEGION_TRITON_CONV2D_H__
#define __LEGION_TRITON_CONV2D_H__

#include "operator.h"
#include "tensor.h"
#ifdef LEGION_USE_CUDA
#include "cudahelp.h"
#endif

namespace triton { namespace backend { namespace legion {

struct Conv2DArgs : public OperatorArgs {
 public:
  Conv2DArgs(bool rel = true, bool bias = true)
      : OperatorArgs(), local_index(0), relu(rel), use_bias(bias)
#ifndef LEGION_USE_CUDA
  {
  }
#else
  {
    workSpace = nullptr;
    workSpaceSize = 0;
  }
  cudnnHandle_t cudnn;
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  void* workSpace;  // device workspace pointer
  size_t workSpaceSize;
#endif
  Legion::Rect<4> input_bounds;
  Legion::Rect<4> local_bounds;
  Legion::Rect<1> bias_bounds;
  DataType input_datatype;
  DataType output_datatype;
  DataType filter_datatype;
  DataType bias_datatype;
  unsigned local_index;
  bool relu, use_bias;
};

class Conv2D : public Operator {
 public:
  Conv2D(
      LegionModelState* model, const LayerStrategy* strategy, size_t inChannels,
      size_t outChannels, size_t kernelH, size_t kernelW, size_t strideH,
      size_t strideW, size_t paddingH, size_t paddingW,
      ActivationMode activation, size_t groups, bool use_bias,
      const char* name);
  virtual ~Conv2D(void);

 public:
  void Configure(
      Tensor* input, Weights* weights, Tensor* output, Weights* bias = NULL);
  Legion::Rect<4> GetInputBounds(Realm::Processor proc);
  Legion::Rect<4> GetWeightBounds(Realm::Processor proc);
  Legion::Rect<1> GetBiasBounds(Realm::Processor proc);
  Legion::Rect<4> GetOutputBounds(Realm::Processor proc);

 public:
  virtual void Load(Realm::Processor processor) override;
  virtual void initialize(
      LegionModelInstance* instance, const unsigned instance_index,
      Legion::Runtime* runtime, Legion::Context ctx,
      Legion::MapperID mapper) override;
  virtual void forward(
      LegionModelInstance* instance, const unsigned instance_index,
      Legion::Runtime* runtime, Legion::Context ctx,
      Legion::MapperID mapper) override;
  virtual void finalize(
      LegionModelInstance* instance, const unsigned instance_index,
      Legion::Runtime* runtime, Legion::Context ctx,
      Legion::MapperID mapper) override;
  virtual void Free(Realm::Processor processor) override;

 public:
  static void PreregisterTaskVariants(void);
  static void forward_cpu(
      const Legion::Task* task,
      const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
      Legion::Runtime* runtime);
#ifdef LEGION_USE_CUDA
 public:
  // Forward task for the GPU
  static void forward_gpu(
      const Legion::Task* task,
      const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
      Legion::Runtime* runtime);

 protected:
  static void forward_kernel(
      const Conv2DArgs* args, const void* input_ptr, void* output_ptr,
      const void* filter_ptr, const void* bias_ptr);
#endif
 public:
  const ActivationMode activation;
  const size_t in_channels, out_channels, kernel_h, kernel_w;
  const size_t stride_h, stride_w, padding_h, padding_w, groups;
  const bool use_bias;

 protected:
  Legion::DomainTransform input_transform;
  Legion::Domain input_extent;
  Conv2DArgs args[MAX_LOCAL_PROCS];
  Legion::FutureMap argmaps[MAX_NUM_INSTANCES];
  Legion::IndexTaskLauncher launchers[MAX_NUM_INSTANCES];
  Legion::ExternalResources weight_attachments[MAX_NUM_INSTANCES];
  Legion::PhysicalRegion bias_attachments[MAX_NUM_INSTANCES];
#ifdef LEGION_USE_CUDA
  void* workspaces[MAX_NUM_INSTANCES][MAX_LOCAL_PROCS];
#endif
};

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_CONV2D_H__
