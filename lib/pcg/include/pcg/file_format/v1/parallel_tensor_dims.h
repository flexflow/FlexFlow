#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_DIMS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_DIMS_H

#include "op-attrs/parallel_tensor_dims.h"
#include "parallel_dim.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ParallelTensorDims {
  req<std::vector<V1ParallelDim>> data;
};
FF_VISITABLE_STRUCT(V1ParallelTensorDims, data);
CHECK_IS_JSONABLE(V1ParallelTensorDims);

V1ParallelTensorDims to_v1(ParallelTensorDims const &dims);
ParallelTensorDims from_v1(V1ParallelTensorDims const &vdims);

} // namespace FlexFlow

#endif
