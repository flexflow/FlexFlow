#ifndef _FLEXFLOW_ELEMENT_BINARY_ATTRS_H
#define _FLEXFLOW_ELEMENT_BINARY_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "core.h"

namespace FlexFlow {

struct ElementBinaryAttrs : use_visitable_cmp<ElementBinaryAttrs> {
public:
  ElementBinaryAttrs() = delete;
  ElementBinaryAttrs(OperatorType, bool should_broadcast_lhs, bool should_broadcast_rhs);
public:
  OperatorType type;
  bool should_broadcast_lhs;
  bool should_broadcast_rhs;
};

}

VISITABLE_STRUCT(::FlexFlow::ElementBinaryAttrs, type);
MAKE_VISIT_HASHABLE(::FlexFlow::ElementBinaryAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<ElementBinaryAttrs>::value, "ElementBinaryAttrs must be a valid opattr (see core.h)");
}

#endif 
