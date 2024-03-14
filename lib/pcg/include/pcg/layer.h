#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_LAYER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_LAYER_H

#include "op-attrs/operator_attrs.h"
#include "utils/stack_string.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct Layer {
public:
  Layer() = delete;
  Layer(CompGraphOperatorAttrs const &attrs, std::optional<std::string> const &name);

public:
  std::optional<stack_string<MAX_OPNAME>> name;
  CompGraphOperatorAttrs attrs;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::Layer, attrs, name);
MAKE_VISIT_HASHABLE(::FlexFlow::Layer);

namespace FlexFlow {

FF_VISIT_FMTABLE(Layer);
// CHECK_FMTABLE(Layer);

} // namespace FlexFlow

#endif
