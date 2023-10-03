#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_SHAPE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_SHAPE_H

#include "datatype.h"
#include "dim_ordered.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

// TODO: V1TensorDims has been commented out because std::vector causes a
// strange error saying it cannot be serialized to JSON and we would like to
// get this to build.
// using V1TensorDims = V1FFOrdered<size_t>;
// CHECK_IS_JSONABLE(V1TensorDims);

struct V1TensorShape {
public:
  // TODO: This should be re-enabled. See the comment above the commented out
  // definition of V1TensorDims.
  // req<V1TensorDims> dims;
  req<V1DataType> data_type;
};
// TODO: This should be re-enabled when the dims field of V1TensorShape is
// re-enabled. See the comment above the commented out definition of
// V1TensorDims.
FF_VISITABLE_STRUCT(V1TensorShape, // dims,
                    data_type);
CHECK_IS_JSONABLE(V1TensorShape);

V1TensorShape to_v1(TensorShape const &t);

} // namespace FlexFlow

#endif
