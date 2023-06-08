#ifndef _FLEXFLOW_SPLIT_ATTRS_H
#define _FLEXFLOW_SPLIT_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct SplitAttrs : public use_visitable_cmp<SplitAttrs> {
public:
  SplitAttrs() = delete;
  SplitAttrs(stack_vector<int, MAX_NUM_OUTPUTS> const &splits, ff_dim_t axis);

public:
  stack_vector<int, MAX_NUM_OUTPUTS> splits;
  ff_dim_t axis;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::SplitAttrs, splits, axis);
MAKE_VISIT_HASHABLE(::FlexFlow::SplitAttrs);

#endif
