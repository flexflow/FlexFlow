#ifndef _FLEXFLOW_CONCAT_ATTRS_H
#define _FLEXFLOW_CONCAT_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

struct ConcatAttrs : use_visitable_cmp<ConcatAttrs> {
public:
  ConcatAttrs() = delete;
  ConcatAttrs(ff_dim_t);

public:
  ff_dim_t axis;
};

}

VISITABLE_STRUCT(::FlexFlow::ConcatAttrs, axis);
MAKE_VISIT_HASHABLE(::FlexFlow::ConcatAttrs);

#endif
