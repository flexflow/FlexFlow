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

#ifndef __LEGION_TRITON_BINARY_H__
#define __LEGION_TRITON_BINARY_H__

#include "operator.h"
#include "tensor.h"

namespace triton {
namespace backend {
namespace legion {

struct BinaryArgs : public OperatorArgs {
public:
  BinaryArgs() = default;
#ifdef LEGION_USE_CUDA
  cudnnHandle_t cudnn;
  cudnnTensorDescriptor_t input0Tensor, input1Tensor, outputTensor;
  cudnnOpTensorDescriptor_t opDesc;
#endif
  OperatorType op_type;
  Legion::Domain bounds;
  DataType datatype;
  bool inplace;
};

class BinaryOperator : public Operator {
public:
  BinaryOperator(LegionModelState *model,
                 LayerStrategy const *strategy,
                 OperatorType type,
                 bool inplace,
                 char const *name);
  virtual ~BinaryOperator() = default;

  void Configure(Tensor *input0, Tensor *input1, Tensor *output);
  Legion::Domain GetBounds(Realm::Processor proc);

  virtual void Load(Realm::Processor processor) override;
  virtual void initialize(LegionModelInstance *instance,
                          unsigned const instance_index,
                          Legion::Runtime *runtime,
                          Legion::Context ctx,
                          Legion::MapperID mapper) override;
  virtual void forward(LegionModelInstance *instance,
                       unsigned const instance_index,
                       Legion::Runtime *runtime,
                       Legion::Context ctx,
                       Legion::MapperID mapper) override;
  virtual void finalize(LegionModelInstance *instance,
                        unsigned const instance_index,
                        Legion::Runtime *runtime,
                        Legion::Context ctx,
                        Legion::MapperID mapper) override;
  virtual void Free(Realm::Processor processor) override;

  static void PreregisterTaskVariants(void);
  static void forward_cpu(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
#ifdef LEGION_USE_CUDA
  static void forward_gpu(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);

protected:
  static bool use_cudnn(OperatorType optype, DataType dtype);
  static void forward_kernel(BinaryArgs const *args,
                             ::cudaStream_t stream,
                             void const *input0_ptr,
                             void const *input1_ptr,
                             void *output_ptr,
                             size_t num_elements);
#endif
public:
  bool const inplace;

protected:
  BinaryArgs args[MAX_LOCAL_PROCS];
  Legion::FutureMap argmaps[MAX_NUM_INSTANCES];
  Legion::IndexTaskLauncher launchers[MAX_NUM_INSTANCES];
};

} // namespace legion
} // namespace backend
} // namespace triton

#endif // __LEGION_TRITON_BINARY_H__
