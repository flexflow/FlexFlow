#ifndef _FLEXFLOW_RUNTIME_SRC_LAYER_H
#define _FLEXFLOW_RUNTIME_SRC_LAYER_H

#include "layer_id.h"
#include "op-attrs/operator_attrs.h"
#include "tensor.h"
#include "utils/optional.h"
#include "utils/stack_string.h"
#include "utils/stack_vector.h"
#include "utils/strong_typedef.h"

namespace FlexFlow {

struct Layer : public use_visitable_cmp<Layer> {
public:
  Layer() = delete;
  Layer(CompGraphOperatorAttrs const &attrs, std::string const &name);

public:
  stack_string<MAX_OPNAME> name;
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
