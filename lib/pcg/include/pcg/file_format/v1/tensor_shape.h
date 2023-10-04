#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_SHAPE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_SHAPE_H

#include "datatype.h"
#include "op-attrs/tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1TensorShape {
public:
  req<std::vector<size_t>> dims;
  req<V1DataType> data_type;
};
FF_VISITABLE_STRUCT(V1TensorShape, dims, data_type);
CHECK_IS_JSONABLE(V1TensorShape);

V1TensorShape to_v1(TensorShape const &t);

} // namespace FlexFlow

#endif
