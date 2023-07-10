#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_OPERATOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_OPERATOR_H

#include "op-attrs/operator_attrs.h"
#include "utils/optional.h"
#include "utils/stack_string.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct Operator : public use_visitable_cmp<Operator> {
public:
  Operator() = delete;
  Operator(PCGOperatorAttrs const &attrs, optional<std::string> const &name);

  operator PCGOperatorAttrs() const;

public:
  PCGOperatorAttrs attrs;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::Operator, attrs);
MAKE_VISIT_HASHABLE(::FlexFlow::Operator);

namespace FlexFlow {
static_assert(is_well_behaved_value_type<Operator>::value, "");
}

#endif
