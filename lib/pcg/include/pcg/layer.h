#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_LAYER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_LAYER_H

#include "op-attrs/operator_attrs.h"
#include "utils/stack_string.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct Layer {
public:
  req<CompGraphOperatorAttrs> attrs;
  req<optional<stack_string<MAX_OPNAME>>> name;
};

FF_VISITABLE_STRUCT(Layer, attrs, name);

} // namespace FlexFlow

namespace FlexFlow {

FF_VISIT_FMTABLE(Layer);
CHECK_FMTABLE(Layer);

} // namespace FlexFlow

#endif
