#ifndef _FLEXFLOW_SOFTMAX_ATTRS_H
#define _FLEXFLOW_SOFTMAX_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "utils/visitable.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

struct SoftmaxAttrs {
public:
  SoftmaxAttrs(ff_dim_t dim);
public:
  ff_dim_t dim;
};

bool operator==(SoftmaxAttrs const &, SoftmaxAttrs const &);
bool operator!=(SoftmaxAttrs const &, SoftmaxAttrs const &);
bool operator<(SoftmaxAttrs const &, SoftmaxAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::SoftmaxAttrs, dim);

namespace std {
template <>
struct hash<::FlexFlow::SoftmaxAttrs> {
  size_t operator()(::FlexFlow::SoftmaxAttrs const &) const;
};
}

#endif
