#ifndef _FLEXFLOW_GROUPBY_ATTRS_H
#define _FLEXFLOW_GROUPBY_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct Group_byAttrs : public use_visitable_cmp<Group_byAttrs> {
public:
  Group_byAttrs(int n, float alpha);

public:
  int n;
  float alpha;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::Group_byAttrs, n, alpha);
MAKE_VISIT_HASHABLE(::FlexFlow::Group_byAttrs);

#endif
