#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_H

#include "data_type.h"
#include "initializer.h"
#include "param_sync.h"
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

struct V1ParallelTensorShape {
  req<std::vector<V1ParallelDim>> dims;
  req<V1DataType> data_type;
};
FF_VISITABLE_STRUCT(V1ParallelTensorShape, dims, data_type);

struct V1ParallelTensor {
  V1ParallelTensorShape shape;
  req<optional<V1ParamSync>> sync_type;
  req<optional<V1Initializer>> initializer;
  req<bool> create_grad;
};
FF_VISITABLE_STRUCT(
    V1ParallelTensor, shape, sync_type, initializer, create_grad);

} // namespace FlexFlow

#endif
