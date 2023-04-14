#ifndef _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H
#define _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H

#include "utils/stack_vector.h"
#include "ffconst.h"
#include "utils/visitable.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/dim_ordered.h"

namespace FlexFlow {

using TensorDims = FFOrdered<size_t>;

struct TensorShape : use_visitable_cmp<TensorShape>, TensorDims {
  TensorShape() = delete;

  template <typename Dims>
  TensorShape(Dims const &dims, DataType data_type)
    : dims(dims), data_type(data_type)
  { }
public:
  TensorDims dims;
  DataType data_type;
};

}

VISITABLE_STRUCT(::FlexFlow::TensorShape, dims, data_type);
MAKE_VISIT_HASHABLE(::FlexFlow::TensorShape);

#endif
