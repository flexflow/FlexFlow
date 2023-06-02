#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_OPERATOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_OPERATOR_H

#include "utils/stack_string.h"
#include "utils/visitable.h"
#include "utils/optional.h"
#include  "op-attrs/operator_attrs.h"

namespace FlexFlow {

struct Operator : public use_visitable_cmp<Operator> {
public:
  Operator() = delete;
  Operator(PCGOperatorAttrs const &attrs,
           optional<std::string> const &name);

  operator PCGOperatorAttrs() const;
public:
  PCGOperatorAttrs attrs;
  optional<stack_string<MAX_OPNAME>> name;
};

}

VISITABLE_STRUCT(::FlexFlow::Operator, attrs, name);
MAKE_VISIT_HASHABLE(::FlexFlow::Operator);

namespace FlexFlow {
static_assert(is_well_behaved_value_type<Operator>::value, "");
}

#endif
