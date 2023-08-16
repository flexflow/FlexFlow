#ifndef _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H
#define _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H

#include "datatype.h"
#include "op-attrs/dim_ordered.h"
#include "op-attrs/ff_dim.h"
#include "utils/stack_vector.h"
#include "utils/visitable.h"

namespace FlexFlow {

using TensorDims = FFOrdered<size_t>;

struct TensorShape {
  size_t at(ff_dim_t) const;
  size_t operator[](ff_dim_t) const;

public:
  req<TensorDims> dims;
  req<DataType> data_type;
};

FF_VISITABLE_STRUCT(TensorShape, dims, data_type);

} // namespace FlexFlow

#endif
