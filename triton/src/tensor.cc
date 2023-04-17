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

#include "tensor.h"
#include "operator.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

Tensor::Tensor(Operator* op, DataType t, const size_t* dims, size_t num_dims)
    : owner(op), type(t), bounds(dims, dims + num_dims)
{
  for (unsigned idx = 0; idx < MAX_NUM_INSTANCES; idx++) {
    region[idx] = LogicalRegion::NO_REGION;
    partition[idx] = LogicalPartition::NO_PART;
  }
}

Tensor::Tensor(Operator* op, DataType t, const std::vector<size_t>& dims)
    : owner(op), type(t), bounds(dims)
{
  for (unsigned idx = 0; idx < MAX_NUM_INSTANCES; idx++) {
    region[idx] = LogicalRegion::NO_REGION;
    partition[idx] = LogicalPartition::NO_PART;
  }
}

Tensor::~Tensor(void) {}

Weights::Weights(Operator* op, DataType t, const size_t* dims, size_t num_dims)
    : Tensor(op, t, dims, num_dims)
{
  const Memory local_sysmem = op->model->runtime_->local_sysmem_;
  for (size_t idx = 0; idx < MAX_LOCAL_PROCS; ++idx) {
    local_memory[idx] = local_sysmem;
    local_allocation[idx] = nullptr;
  }
}

Weights::Weights(Operator* op, DataType t, const std::vector<size_t>& dims)
    : Tensor(op, t, dims)
{
  const Memory local_sysmem = op->model->runtime_->local_sysmem_;
  for (size_t idx = 0; idx < MAX_LOCAL_PROCS; ++idx) {
    local_memory[idx] = local_sysmem;
    local_allocation[idx] = nullptr;
  }
}

Weights::~Weights(void)
{
  for (size_t idx = 0; idx < MAX_LOCAL_PROCS; ++idx) {
    assert(local_allocation[idx] == nullptr);
  }
}

}}}  // namespace triton::backend::legion
