#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_H

#include "datatype.h"
#include "initializer.h"
#include "param_sync.h"
#include "pcg/parallel_tensor.h"
#include "utils/json.h"
#include "utils/variant.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ParallelDim {
  req<size_t> size;
  req<int> degree;
  req<bool> is_replica_dim;
};

FF_VISITABLE_STRUCT(V1ParallelDim, size, degree, is_replica_dim);
CHECK_IS_JSONABLE(V1ParallelDim);
V1ParallelDim to_v1(ParallelDim const &);

struct V1ParallelTensorShape {
  req<std::vector<V1ParallelDim>> dims;
  req<V1DataType> data_type;
};

FF_VISITABLE_STRUCT(V1ParallelTensorShape, dims, data_type);
CHECK_IS_JSONABLE(V1ParallelTensorShape);
V1ParallelTensorShape to_v1(ParallelTensorShape const &);

struct V1ParallelTensor {
  V1ParallelTensorShape shape;
  req<bool> create_gradients;
  req<optional<V1Initializer>> initializer;
  req<optional<V1ParamSync>> sync_type;
  req<optional<std::string>> name;
};

FF_VISITABLE_STRUCT(
    V1ParallelTensor, shape, create_gradients, initializer, sync_type, name);
CHECK_IS_JSONABLE(V1ParallelTensor);
V1ParallelTensor to_v1(ParallelTensor const &);

} // namespace FlexFlow

#endif
