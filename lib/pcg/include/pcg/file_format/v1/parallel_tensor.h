#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_H

#include "create_grad.h"
#include "datatype.h"
#include "initializer.h"
#include "parallel_tensor_dims.h"
#include "parallel_tensor_shape.h"
#include "param_sync.h"
#include "pcg/parallel_tensor.h"
#include "utils/json.h"
#include "utils/variant.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ParallelTensor {
  V1ParallelTensorShape shape;
  req<V1CreateGrad> create_gradients;
  req<optional<V1Initializer>> initializer;
  req<optional<V1ParamSync>> sync_type;
  req<optional<std::string>> name;
};
FF_VISITABLE_STRUCT(
    V1ParallelTensor, shape, create_gradients, initializer, sync_type, name);
CHECK_IS_JSONABLE(V1ParallelTensor);

V1ParallelTensor to_v1(ParallelTensor const &t);
ParallelTensor from_v1(V1ParallelTensor const &vt);

} // namespace FlexFlow

#endif
