#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_SHAPE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_SHAPE_H

#include "datatype.h"
#include "dim_ordered.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

// using V1TensorDims = V1FFOrdered<size_t>;
// CHECK_IS_JSONABLE(V1TensorDims);

struct V1TensorShape {
public:
  // req<V1TensorDims> dims;
  req<V1DataType> data_type;
};
FF_VISITABLE_STRUCT(V1TensorShape, // dims,
                    data_type);
CHECK_IS_JSONABLE(V1TensorShape);

V1TensorShape to_v1(TensorShape const &t);

} // namespace FlexFlow

#endif
