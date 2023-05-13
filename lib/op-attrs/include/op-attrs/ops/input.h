#ifndef _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H
#define _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H

#include "utils/visitable.h"
#include "core.h"

namespace FlexFlow {

struct InputAttrs : public use_visitable_cmp<InputAttrs> { 
public:
  InputAttrs() = default;
};

}

VISITABLE_STRUCT_EMPTY(::FlexFlow::InputAttrs);
MAKE_VISIT_HASHABLE(::FlexFlow::InputAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<InputAttrs>::value, "");
}

#endif
