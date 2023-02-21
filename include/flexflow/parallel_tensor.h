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

#pragma once

#include "flexflow/ffconst.h"
#include "flexflow/machine_view.h"
#include "flexflow/utils/dot/record_formatter.h"
#include "legion.h"
#include <ostream>
#include <unordered_map>

namespace FlexFlow {

class Op;
class FFModel;
class Initializer;

struct ParallelDim {
  static constexpr int UNKNOWN_DEGREE = -1;
  static constexpr int UNKNOWN_INDEX = -2;

  bool operator==(ParallelDim const &rhs) const {
    if (size != rhs.size) {
      return false;
    }
    if (degree != rhs.degree) {
      return false;
    }
    if (parallel_idx != rhs.parallel_idx) {
      return false;
    }
    return true;
  }

  bool operator!=(ParallelDim const &rhs) const {
    if (size != rhs.size) {
      return true;
    }
    if (degree != rhs.degree) {
      return true;
    }
    if (parallel_idx != rhs.parallel_idx) {
      return true;
    }
    return false;
  }

  int size = 0;                     // Actual size of tensor
  int degree = UNKNOWN_DEGREE;      // Degree of sharding
  int parallel_idx = UNKNOWN_INDEX; // Runtime information, unique id of each
                                    // degree of sharding
  bool is_replica_dim = false;
};

/**
 * @brief Represent the shape of a ParallelTensor.
 */
struct ParallelTensorShape {

  /**
   * @brief Default constructor.
   */
  ParallelTensorShape() = default;

  /**
   * @brief Construct a new ParallelTensorShape object.
   *
   * @param num_dims Number of dimensions
   * @param dims Details of each dimension
   * @param data_type The data type
   */
  ParallelTensorShape(int num_dims,
                      ParallelDim const dims[MAX_TENSOR_DIM],
                      DataType data_type);

  int num_dims;                     ///< Number of dimensions
  ParallelDim dims[MAX_TENSOR_DIM]; ///< Details of each dimension
  DataType data_type;               ///< Data type

  bool operator==(ParallelTensorShape const &other) const;
  bool operator!=(ParallelTensorShape const &other) const;

  RecordFormatter as_dot() const;

  size_t get_piece_size() const;
  bool is_valid() const;

  int get_num_replica_dims() const;
  int get_num_replicas() const;

  std::unordered_map<int, int> get_mv_dim_to_tensor_dim_mapping() const;
  std::unordered_map<int, int> get_tensor_dim_to_mv_dim_mapping() const;
};

std::ostream &operator<<(std::ostream &, ParallelTensorShape const &);

}; // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ParallelTensorShape> {
  size_t operator()(FlexFlow::ParallelTensorShape const &) const;
};
} // namespace std

namespace FlexFlow {

class FFConfig;

/**
 * @brief Base structure of the parallel tensor representation.
 *
 * @details Parallel tensor is the fundamental component to support the
 * representation and exploration of parallelization strategies.
 */
struct ParallelTensorBase {
  static constexpr ParallelTensorBase *NO_TENSOR = nullptr;
  ParallelTensorBase(void) = default;
  ParallelTensorBase(ParallelTensorBase const &rhs);
  // Tensor& operator=(const Tensor& rhs);
  // bool operator==(const Tensor& rhs) const;
  void inline_map(FFConfig &config);
  void inline_unmap(FFConfig &config);
  template <typename T>
  T *get_raw_ptr(FFConfig &config);
  void attach_raw_ptr(FFConfig &config, void *raw_ptr, bool column_major);
  void detach_raw_ptr(FFConfig &config);
  bool get_input_sub_tensor(ParallelConfig const &pc,
                            ParallelTensorBase &tensor,
                            OperatorType type);
  bool get_sub_tensor(MachineView const &mv,
                      ParallelTensorBase &subtensor) const;
  bool get_output_sub_tensor(ParallelConfig const &pc,
                             ParallelTensorBase &tensor,
                             OperatorType type);
  size_t get_owner_independent_hash() const;
  size_t get_volume() const;
  size_t get_total_num_parts() const;
  int get_num_replica_dims() const;
  int get_num_replicas() const;
  Legion::Domain get_domain() const;
  bool check_valid() const;
  bool is_valid_machine_view(MachineView const &view) const;
  void print(std::string const &name) const;
  static bool update_parallel_ids(int numdim, ParallelDim *dims);
  template <typename T>
  bool set_tensor(FFModel const *model,
                  std::vector<int> const &dims,
                  T const *data);
  template <typename T>
  bool get_tensor(FFModel const *model, T *data, bool get_parameters);
  ParallelTensorShape get_shape() const;

private:
  template <typename T>
  bool get_input_sub_tensor_via_mappings(ParallelConfig const &pc,
                                         ParallelTensorBase &tensor) const;

public:
  size_t parallel_tensor_guid = 0;
  int num_dims = 0;
  // int adim[MAX_TENSOR_DIM];
  ParallelDim dims[MAX_TENSOR_DIM];
  DataType data_type = DT_NONE;
  ParameterSyncType sync_type = ParameterSyncType::NONE;
  Initializer *initializer = nullptr;
  // Describes the ownership of this tensor
  Op const *owner_op = nullptr;
  int owner_idx = 0;
  bool create_gradients = false;

  // The following fields are initialized after model.compile
  MachineView machine_view = MachineView::NO_VIEW;
  Legion::IndexSpace parallel_is = Legion::IndexSpace::NO_SPACE;
  Legion::LogicalRegion region = Legion::LogicalRegion::NO_REGION,
                        region_grad = Legion::LogicalRegion::NO_REGION;
  Legion::LogicalPartition part = Legion::LogicalPartition::NO_PART,
                           part_grad = Legion::LogicalPartition::NO_PART;
  Legion::PhysicalRegion physical_region;
};

typedef ParallelTensorBase *ParallelTensor;
typedef ParallelTensorBase *ParallelParameter;

}; // namespace FlexFlow
