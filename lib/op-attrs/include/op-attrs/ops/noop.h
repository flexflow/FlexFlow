#ifndef _FLEXFLOW_OP_ATTRS_OPS_NOOP_H
#define _FLEXFLOW_OP_ATTRS_OPS_NOOP_H

#include "utils/visitable.h"
#include <functional>

namespace FlexFlow {

struct NoopAttrs : public use_visitable_cmp<NoopAttrs> { 
public:
  NoopAttrs() = default;
};

struct InputAttrs : public use_visitable_cmp<InputAttrs> { 
public:
  InputAttrs() = default;
};

}

VISITABLE_STRUCT_EMPTY(::FlexFlow::NoopAttrs);
VISITABLE_STRUCT_EMPTY(::FlexFlow::InputAttrs);
MAKE_VISIT_HASHABLE(::FlexFlow::NoopAttrs);
MAKE_VISIT_HASHABLE(::FlexFlow::InputAttrs);

#endif
