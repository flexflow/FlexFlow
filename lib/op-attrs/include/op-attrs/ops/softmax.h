#ifndef _FLEXFLOW_SOFTMAX_ATTRS_H
#define _FLEXFLOW_SOFTMAX_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct SoftmaxAttrs : public use_visitable_cmp<SoftmaxAttrs> {
public:
  SoftmaxAttrs() = delete;
  explicit SoftmaxAttrs(ff_dim_t dim);

public:
  ff_dim_t dim;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::SoftmaxAttrs, dim);
MAKE_VISIT_HASHABLE(::FlexFlow::SoftmaxAttrs);

#endif
