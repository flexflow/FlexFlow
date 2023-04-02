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

#ifndef _FLEXFLOW_RUNTIME_SRC_TENSOR_H
#define _FLEXFLOW_RUNTIME_SRC_TENSOR_H

#include "legion.h"
#include <unordered_map>
#include <memory>
#include "op-attrs/ffconst.h"
#include "utils/stack_vector.h"
#include "kernels/array_shape.h"
#include "parallel_tensor.h"
#include "tensor_shape.h"
#include <type_traits>

namespace FlexFlow {

class Layer;
class FFModel;
class Initializer;

struct TensorBase {
  TensorBase(void) = default;
  TensorBase(TensorBase const &rhs);
  TensorBase(size_t tensor_guid, 
             TensorShape const &,
             bool create_gradients, 
             Initializer const *initializer = nullptr, 
             ParameterSyncType sync_type = ParameterSyncType::NONE);

  size_t get_volume() const;
  Legion::Domain get_domain() const;

  int num_dims() const;

  void print(std::string const &name) const;
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
public:
  size_t tensor_guid = 0;
  // int adim[MAX_TENSOR_DIM];
  stack_vector<int, MAX_TENSOR_DIM> dims;
  DataType data_type = DT_NONE;
  ParameterSyncType sync_type = ParameterSyncType::NONE;
  Initializer *initializer = nullptr;
  optional<ParallelTensor> parallel_tensor = nullopt;
  // Describes the ownership of this tensor
  Layer const *owner_layer = nullptr;
  int owner_idx = 0;
  bool create_gradients = false;
};

struct Tensor {
public: 
  Tensor() = delete;
  explicit Tensor(std::shared_ptr<TensorBase const> ptr);

  template <typename ...Args>
  Tensor(Args&&...args)
    : ptr(std::make_shared<TensorBase>(std::forward<Args>(args)...))
  { }

  Tensor(Tensor const &) = default;
  Tensor(Tensor &) = default;

  TensorBase *operator->();
  TensorBase const *operator->() const;
private:
  std::shared_ptr<TensorBase const> ptr;
};

static_assert(std::is_copy_constructible<Tensor>::value, "Tensor must be copy constructible");

using Parameter = Tensor;

}

#endif
