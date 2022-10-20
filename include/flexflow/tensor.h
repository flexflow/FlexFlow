/* Copyright 2021 CMU, Facebook, LANL, MIT, and Stanford (alphabetical)
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

#ifndef _FLEXFLOW_TENSOR_H
#define _FLEXFLOW_TENSOR_H

#include "flexflow/parallel_tensor.h"
#include "flexflow/machine_view.h"
#include "legion.h"
#include <unordered_map>

namespace FlexFlow {

class Layer;
class FFModel;
class Initializer;

struct TensorBase {
  TensorBase(void) = default;
  TensorBase(TensorBase const &rhs);

  size_t get_volume() const;

  Legion::Domain get_domain() const;

  void print(std::string const &name) const;

  template <typename T>
  bool set_tensor(FFModel const *model,
                  std::vector<int> const &dims,
                  const T *data);
  template <typename T>
  bool get_tensor(FFModel const *model, T *data, bool get_gradients);
public:
  size_t tensor_guid = 0;
  int num_dims = 0;
  int dims[MAX_TENSOR_DIM];
  DataType data_type = DT_NONE;
  ParameterSyncType sync_type = ParameterSyncType::NONE;
  Initializer *initializer = nullptr;
  ParallelTensor parallel_tensor;
  Layer const *owner_layer = nullptr;  ///< Describes the ownership of this tensor
  int owner_idx = 0;
  bool create_gradients = false;
};

typedef TensorBase *Tensor;
typedef TensorBase *Parameter;

}; // namespace FlexFlow

#endif // _FLEXFLOW_TENSOR_H
