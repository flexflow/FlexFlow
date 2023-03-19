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

#ifndef __LEGION_TRITON_UNARY_H__
#define __LEGION_TRITON_UNARY_H__

#include "operator.h"
#include "tensor.h"

namespace triton { namespace backend { namespace legion {

class UnaryOperator;

struct UnaryArgs : public OperatorArgs {
 public:
  UnaryArgs(void);
#ifdef LEGION_USE_CUDA
  cudnnHandle_t cudnn;
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#endif
  unsigned local_index;
  OperatorType op_type;
  Legion::Domain bounds;
  DataType datatype;
  DataType casttype;
  union {
    int8_t int8_value;
    __half half_value;
    float float_value;
    double double_value;
  } scalar;
  bool inplace;
};

class UnaryOperator : public Operator {
 public:
  UnaryOperator(
      LegionModelState* model, const LayerStrategy* strategy, OperatorType type,
      const void* scalar, DataType scalar_type, bool inplace, const char* name);
  virtual ~UnaryOperator(void);

  void Configure(Tensor* input, Tensor* output);
  Legion::Domain GetBounds(Realm::Processor proc);

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
  static void forward_gpu(
      const Legion::Task* task,
      const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
      Legion::Runtime* runtime);

 protected:
  static bool use_cudnn(OperatorType optype);
  static void forward_kernel(
      const UnaryArgs* args, ::cudaStream_t stream, const void* input_ptr,
      void* output_ptr, size_t num_elements);
#endif
 public:
  const DataType scalar_type;
  union {
    int8_t int8_value;
    __half half_value;
    float float_value;
    double double_value;
  } scalar;
  const bool inplace;

 protected:
  UnaryArgs args[MAX_LOCAL_PROCS];
  Legion::FutureMap argmaps[MAX_NUM_INSTANCES];
  Legion::IndexTaskLauncher launchers[MAX_NUM_INSTANCES];
};

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_UNARY_H__
