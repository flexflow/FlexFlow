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

#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_H

#include "create_grad.h"
#include "initializer.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "parallel_tensor_guid_t.h"
#include "pcg/machine_view.h"
#include "utils/optional.h"
#include "utils/record_formatter.h"
#include "utils/strong_typedef.h"
#include <ostream>
#include <unordered_map>

namespace FlexFlow {

/**
 * @brief Base structure of the parallel tensor representation.
 *
 * @details Parallel tensor is the fundamental component to support the
 * representation and exploration of parallelization strategies.
 */
struct ParallelTensorAttrs : public use_visitable_cmp<ParallelTensorAttrs> {
  ParallelTensorAttrs() = delete;

  ParallelTensorAttrs(ParallelTensorShape const &,
                      CreateGrad create_gradients,
                      optional<ParamSync> sync_type = nullopt,
                      optional<Initializer> initializer = nullopt);
  ParallelTensorAttrs(ParallelTensorDims const &,
                      DataType,
                      CreateGrad create_gradients,
                      optional<ParamSync> sync_type = nullopt,
                      optional<Initializer> initializer = nullopt);

  /* void inline_map(FFConfig &config); */
  /* void inline_unmap(FFConfig &config); */
  /* template <typename T> */
  /* T *get_raw_ptr(FFConfig &config); */
  /* void attach_raw_ptr(FFConfig &config, void *raw_ptr, bool column_major); */
  /* void detach_raw_ptr(FFConfig &config); */
  /* bool get_input_sub_tensor(MachineView const &, */
  /*                           ParallelTensor &tensor, */
  /*                           OperatorType type); */
  /* bool get_sub_tensor(MachineView const &mv, */
  /*                     ParallelTensor &subtensor) const; */
  /* bool get_output_sub_tensor(MachineView const &, */
  /*                            ParallelTensor &tensor, */
  /*                            OperatorType type); */
  /* size_t get_owner_independent_hash() const; */
  /* size_t get_volume() const; */
  /* size_t get_total_num_parts() const; */
  /* int get_num_replica_dims() const; */
  /* int get_num_replicas() const; */
  /* /1* Legion::Domain get_domain() const; *1/ */
  /* bool check_valid() const; */
  /* bool is_valid_machine_view(MachineView const &view) const; */
  /* void print(std::string const &name) const; */
  /* static bool update_parallel_ids(int numdim, ParallelDim *dims); */
  /* ParallelTensorShape get_shape() const; */

  /* private: */
  /* template <typename T> */
  /* bool get_input_sub_tensor_via_mappings(MachineView const &, */
  /*                                        ParallelTensor &tensor) const; */

public:
  ParallelTensorDims dims;
  DataType data_type;
  optional<ParamSync> sync_type = nullopt;
  optional<Initializer> initializer = nullopt;
  CreateGrad create_gradients;
};

using ParallelParameter = ParallelTensor;

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::ParallelTensorAttrs,
                 dims,
                 data_type,
                 sync_type,
                 initializer,
                 create_gradients);

namespace FlexFlow {
static_assert(is_well_behaved_value_type<ParallelTensorAttrs>::value, "");
}

#endif
