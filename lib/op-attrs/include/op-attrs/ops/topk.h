#ifndef _FLEXFLOW_TOPK_ATTRS_H
#define _FLEXFLOW_TOPK_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct TopKAttrs : public use_visitable_cmp<TopKAttrs> {
public:
  TopKAttrs() = delete;
  TopKAttrs(int k, bool sorted);

public:
  int k;
  bool sorted;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::TopKAttrs, k, sorted);
MAKE_VISIT_HASHABLE(::FlexFlow::TopKAttrs);

#endif
