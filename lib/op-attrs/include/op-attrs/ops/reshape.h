#ifndef _FLEXFLOW_RESHAPE_ATTRS_H
#define _FLEXFLOW_RESHAPE_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ReshapeAttrs {
  stack_vector<int, MAX_TENSOR_DIM> shape;
};

bool operator==(ReshapeAttrs const &, ReshapeAttrs const &);
bool operator<(ReshapeAttrs const &, ReshapeAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::ReshapeAttrs, shape);

namespace std {
template <>
struct hash<::FlexFlow::ReshapeAttrs> {
  size_t operator()(::FlexFlow::ReshapeAttrs const &) const;
};
}

#endif 
