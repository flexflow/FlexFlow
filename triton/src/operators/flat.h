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

#ifndef __LEGION_TRITON_FLAT_H__
#define __LEGION_TRITON_FLAT_H__

#include "operator.h"
#include "tensor.h"

namespace triton { namespace backend { namespace legion {

struct FlatArgs : public OperatorArgs {
 public:
};

class Flat : public Operator {
 public:
  Flat(LegionModelState* state, const char* name);

  void configure(Tensor* input, Tensor* output);

  virtual void initialize(LegionModelInstance* instance);
  virtual void forward(LegionModelInstance* instance);
  virtual void finalize(LegionModelInstance* instance);

  static FlatArgs initalize_gpu(
      const Legion::Task* task,
      const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
      Legion::Runtime* runtime);
  static void forward_gpu(
      const Legion::Task* task,
      const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
      Legion::Runtime* runtime);
  static void forward_kernel(
      const FlatArgs* args, const void* input_ptr, void* output_ptr,
      size_t num_elements);

 public:
  LegionModelState* const model;
};

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_FLAT_H__
