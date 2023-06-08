/**
 * @file parallel_tensor.h
 * @brief Parallel Tensor Representation
 *
 * @copyright Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford
 * (alphabetical)
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

#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_TENSOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_TENSOR_H

#include "create_grad.h"
#include "initializer.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/param_sync.h"

namespace FlexFlow {

/**
 * @brief Base structure of the parallel tensor representation.
 *
 * @details Parallel tensor is the fundamental component to support the
 * representation and exploration of parallelization strategies.
 */
struct ParallelTensor : public use_visitable_cmp<ParallelTensor> {
  ParallelTensor() = delete;

  ParallelTensor(ParallelTensorShape const &,
                 CreateGrad create_gradients,
                 optional<ParamSync> sync_type = nullopt,
                 optional<Initializer> initializer = nullopt);
  ParallelTensor(ParallelTensorDims const &,
                 DataType,
                 CreateGrad create_gradients,
                 optional<ParamSync> sync_type = nullopt,
                 optional<Initializer> initializer = nullopt);

public:
  ParallelTensorDims dims;
  DataType data_type;
  optional<ParamSync> sync_type = nullopt;
  optional<Initializer> initializer = nullopt;
  CreateGrad create_gradients;
};

using ParallelParameter = ParallelTensor;

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::ParallelTensor,
                 dims,
                 data_type,
                 sync_type,
                 initializer,
                 create_gradients);
MAKE_VISIT_HASHABLE(::FlexFlow::ParallelTensor);

namespace FlexFlow {
static_assert(is_well_behaved_value_type<ParallelTensor>::value, "");
}

#endif
