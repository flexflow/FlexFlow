#ifndef _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H
#define _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H

#include "datatype.h"
#include "op-attrs/dim_ordered.h"
#include "op-attrs/ff_dim.h"
#include "utils/stack_vector.h"
#include "utils/visitable.h"

namespace FlexFlow {

using TensorDims = FFOrdered<size_t>;

struct TensorShape : public use_visitable_cmp<TensorShape> {
  TensorShape() = delete;

  template <typename Dims>
  TensorShape(Dims const &dims, DataType data_type)
      : dims(dims), data_type(data_type) {}

  size_t at(ff_dim_t) const;
  size_t operator[](ff_dim_t) const;

public:
  TensorDims dims;
  DataType data_type;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::TensorShape, dims, data_type);
MAKE_VISIT_HASHABLE(::FlexFlow::TensorShape);

#endif
