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

#include "create_grad.h"
#include "initializer.h"
/* #include "kernels/array_shape.h" */
/* #include "legion.h" */
/* #include "legion_tensor_shape.h" */
/* #include "op-attrs/datatype.h" */
/* #include "op-attrs/param_sync.h" */
/* #include "op-attrs/tensor_shape.h" */
/* #include "utils/optional.h" */
/* #include "utils/stack_vector.h" */
/* #include "utils/visitable.h" */
/* #include <memory> */
/* #include <type_traits> */
/* #include <unordered_map> */

/* namespace FlexFlow { */

/* struct Tensor { */
/*   Tensor() = delete; */
/*   Tensor(TensorShape const &, */
/*          CreateGrad create_gradients, */
/*          optional<Initializer> const &initializer = nullopt, */
/*          optional<ParamSync> sync_type = nullopt); */

/*   size_t get_volume() const; */
/*   Legion::Domain get_domain() const; */
/*   TensorShape get_shape() const; */
/*   int num_dims() const; */

/*   operator TensorShape const &() const; */
/* public: */
/*   TensorShape shape; */
/*   CreateGrad create_gradients; */
/*   optional<Initializer> initializer; */
/*   optional<ParamSync> sync_type; */
/* }; */
/* FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Tensor, shape, create_gradients,
 * initializer, sync_type); */

/* template <typename T> */
/* bool set_tensor(Tensor const &, */
/*                 FFModel const *model, */
/*                 std::vector<int> const &dims, */
/*                 T const *data); */
/* template <typename T> */
/* bool get_tensor(Tensor const &, */
/*                 FFModel const *model, */
/*                 T *data, */
/*                 bool get_gradients); */

/* using Parameter = Tensor; */

} // namespace FlexFlow

#endif
