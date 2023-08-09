#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_LAYER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_LAYER_H

#include "op-attrs/operator_attrs.h"
#include "utils/stack_string.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct Layer : public use_visitable_cmp<Layer> {
public:
  Layer() = delete;
  Layer(CompGraphOperatorAttrs const &attrs, optional<std::string> const &name);

public:
  optional<stack_string<MAX_OPNAME>> name;
  CompGraphOperatorAttrs attrs;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::Layer, attrs, name);
MAKE_VISIT_HASHABLE(::FlexFlow::Layer);

namespace FlexFlow {
static_assert(is_well_behaved_value_type<Layer>::value, "");
static_assert(is_fmtable<Layer>::value, "Layer must be fmtable");
} // namespace FlexFlow

#endif
