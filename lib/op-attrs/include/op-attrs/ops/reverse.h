#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REVERSE_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REVERSE_H

#include "utils/visitable.h"
#include "core.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

struct ReverseAttrs : public use_visitable_cmp<ReverseAttrs> {
public:
  ReverseAttrs() = delete;
  explicit ReverseAttrs(ff_dim_t const &axis);
public:
  ff_dim_t axis;
};

}

VISITABLE_STRUCT(::FlexFlow::ReverseAttrs, axis);
MAKE_VISIT_HASHABLE(::FlexFlow::ReverseAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<ReverseAttrs>::value, "");
}

#endif
