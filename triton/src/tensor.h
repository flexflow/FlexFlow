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

#ifndef __LEGION_TRITON_TENSOR_H__
#define __LEGION_TRITON_TENSOR_H__

#include "config.h"
#include "legion.h"
#include "types.h"

namespace triton { namespace backend { namespace legion {

class Tensor {
 public:
  Tensor(Operator* op, DataType type, const size_t* dims, size_t num_dims);
  Tensor(Operator* op, DataType type, const std::vector<size_t>& dims);
  virtual ~Tensor(void);

 public:
  Operator* const owner;
  const DataType type;
  const std::vector<size_t> bounds;

 public:
  Legion::LogicalRegion region[MAX_NUM_INSTANCES];
  Legion::LogicalPartition partition[MAX_NUM_INSTANCES];
};

class Weights : public Tensor {
 public:
  Weights(Operator* op, DataType type, const size_t* dims, size_t num_dims);
  Weights(Operator* op, DataType type, const std::vector<size_t>& dims);
  virtual ~Weights(void);

 public:
  Legion::Domain local_bounds[MAX_LOCAL_PROCS];
  Legion::Memory local_memory[MAX_LOCAL_PROCS];
  void* local_allocation[MAX_LOCAL_PROCS];
  size_t local_strides[MAX_LOCAL_PROCS][LEGION_MAX_DIM];
};

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_TENSOR_H__
