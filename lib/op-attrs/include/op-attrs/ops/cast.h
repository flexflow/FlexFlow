#ifndef _FLEXFLOW_CAST_ATTRS_H
#define _FLEXFLOW_CAST_ATTRS_H

#include "core.h"
#include "op-attrs/datatype.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct CastAttrs {
  req<DataType> dtype;
};
FF_VISITABLE_STRUCT(CastAttrs, dtype);

ParallelTensorShape get_output_shape(CastAttrs const &,
                                     ParallelTensorShape const &);

CHECK_VALID_OP_ATTR(CastAttrs);
} // namespace FlexFlow

#endif
