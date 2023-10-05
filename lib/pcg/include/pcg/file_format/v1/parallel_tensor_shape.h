#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_SHAPE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_SHAPE_H

#include "datatype.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "parallel_tensor_dims.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ParallelTensorShape {
  V1ParallelTensorDims dims;
  req<V1DataType> data_type;
};
FF_VISITABLE_STRUCT(V1ParallelTensorShape, dims, data_type);
CHECK_IS_JSONABLE(V1ParallelTensorShape);

V1ParallelTensorShape to_v1(ParallelTensorShape const &t);
ParallelTensorShape from_v1(V1ParallelTensorShape const &vt);

} // namespace FlexFlow

#endif
