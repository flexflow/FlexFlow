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

#ifndef __LEGION_TRITON_LINEAR_H__
#define __LEGION_TRITON_LINEAR_H__

#include "operator.h"
#include "tensor.h"

namespace triton { namespace backend { namespace legion {

struct LinearArgs : public OperatorArgs {
 public:
  LinearArgs(size_t batch_size);
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  ActivationMode activation;
  bool use_bias;
};

class Linear : public Operator {
 public:
  Linear(
      LegionModelState* model, unsigned out_dim, ActivationMode activation,
      bool use_bias, const char* name);

  void configure(
      Tensor* input, Weights* weights, Tensor* output, Weights* bias = NULL);

  virtual void initialize(LegionModelInstance* instance);
  virtual void forward(LegionModelInstance* instance);
  virtual void finalize(LegionModelInstance* instance);

  static LinearArgs initialize_gpu(
      const Legion::Task* task,
      const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
      Legion::Runtime* runtime);
  static void forward_gpu(
      const Legion::Task* task,
      const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
      Legion::Runtime* runtime);
  static void forward_kernel(
      const LinearArgs* args, const void* input_ptr, void* output_ptr,
      const void* filter_ptr, const void* bias_ptr, unsigned in_dim,
      unsigned out_dim, size_t batch_size);

 public:
  LegionModelState* const model;
  const unsigned in_channels, out_channels;
  const bool use_bias;
};

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_LINEAR_H__
