#ifndef _FLEXFLOW_OP_ATTRS_OPS_NOOP_H
#define _FLEXFLOW_OP_ATTRS_OPS_NOOP_H

#include "core.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct NoopAttrs : public use_visitable_cmp<NoopAttrs> {
public:
  NoopAttrs() = default;
};

} // namespace FlexFlow

VISITABLE_STRUCT_EMPTY(::FlexFlow::NoopAttrs);
MAKE_VISIT_HASHABLE(::FlexFlow::NoopAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<NoopAttrs>::value, "");
}

#endif
