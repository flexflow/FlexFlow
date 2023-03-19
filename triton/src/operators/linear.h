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

namespace triton {
namespace backend {
namespace legion {

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
  Linear(LegionModelState *model,
         unsigned out_dim,
         ActivationMode activation,
         bool use_bias,
         char const *name);

  void configure(Tensor *input,
                 Weights *weights,
                 Tensor *output,
                 Weights *bias = NULL);

  virtual void initialize(LegionModelInstance *instance);
  virtual void forward(LegionModelInstance *instance);
  virtual void finalize(LegionModelInstance *instance);

  static LinearArgs
      initialize_gpu(Legion::Task const *task,
                     std::vector<Legion::PhysicalRegion> const &regions,
                     Legion::Context ctx,
                     Legion::Runtime *runtime);
  static void forward_gpu(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
  static void forward_kernel(LinearArgs const *args,
                             void const *input_ptr,
                             void *output_ptr,
                             void const *filter_ptr,
                             void const *bias_ptr,
                             unsigned in_dim,
                             unsigned out_dim,
                             size_t batch_size);

public:
  LegionModelState *const model;
  unsigned const in_channels, out_channels;
  bool const use_bias;
};

} // namespace legion
} // namespace backend
} // namespace triton

#endif // __LEGION_TRITON_LINEAR_H__
