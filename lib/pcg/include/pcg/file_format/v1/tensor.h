#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_H

#include "create_grad.h"
#include "initializer.h"
#include "param_sync.h"
#include "pcg/tensor.h"
#include "tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1Tensor {
  V1TensorShape shape;
  req<V1CreateGrad> create_gradients;
  req<optional<V1Initializer>> initializer;
  req<optional<V1ParamSync>> sync_type;
  req<optional<std::string>> name;
};
FF_VISITABLE_STRUCT(
    V1Tensor, shape, create_gradients, initializer, sync_type, name);
CHECK_IS_JSONABLE(V1Tensor);

V1Tensor to_v1(Tensor const &);
Tensor from_v1(V1Tensor const &);

} // namespace FlexFlow

#endif
