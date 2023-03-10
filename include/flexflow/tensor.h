/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "flexflow/machine_view.h"
#include "legion.h"
#include <unordered_map>

namespace FlexFlow {

class Layer;
class FFModel;
class Initializer;
class ParallelTensorBase;

struct TensorBase {
  TensorBase(void) = default;
  TensorBase(TensorBase const &rhs);
  // void inline_map(FFConfig &config);
  // void inline_unmap(FFConfig &config);
  // template<typename T>
  // T* get_raw_ptr(FFConfig &config);
  // void attach_raw_ptr(FFConfig &config, void *raw_ptr, bool column_major);
  // void detach_raw_ptr(FFConfig &config);
  // bool get_input_sub_tensor(const ParallelConfig& pc,
  //                           TensorBase& tensor,
  //                           OperatorType type);
  // bool get_output_sub_tensor(const ParallelConfig& pc,
  //                            TensorBase& tensor,
  //                            OperatorType type);
  // size_t get_owner_independent_hash() const;
  size_t get_volume() const;
  // size_t get_total_num_parts() const;
  Legion::Domain get_domain() const;
  // bool check_valid() const;
  // bool is_valid_machine_view(const MachineView& view) const;
  void print(std::string const &name) const;
  // static bool update_parallel_ids(int numdim, ParallelDim* dims);
  template <typename T>
  bool set_tensor(FFModel const *model,
                  std::vector<int> const &dims,
                  T const *data);
  template <typename T>
  bool get_tensor(FFModel const *model, T *data, bool get_gradients);
  template <typename T>
  bool get_output_parallel_tensor(FFModel const *ff,
                                  T *data,
                                  bool get_gradients);
  // TensorShape get_shape() const;
private:
  // template <typename T>
  // bool get_input_sub_tensor_via_mappings(const ParallelConfig& pc,
  // TensorBase& tensor) const;
public:
  size_t tensor_guid = 0;
  int num_dims = 0;
  // int adim[MAX_TENSOR_DIM];
  int dims[MAX_TENSOR_DIM];
  DataType data_type = DT_NONE;
  ParameterSyncType sync_type = ParameterSyncType::NONE;
  Initializer *initializer = nullptr;
  ParallelTensorBase *parallel_tensor = nullptr;
  // Describes the ownership of this tensor
  Layer const *owner_layer = nullptr;
  int owner_idx = 0;
  bool create_gradients = false;
};

typedef TensorBase *Tensor;
typedef TensorBase *Parameter;

}; // namespace FlexFlow
