#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_OPERATOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_OPERATOR_H

#include "op-attrs/operator_attrs.h"
#include "utils/stack_string.h"
#include "utils/visitable.h"

#include <optional>

namespace FlexFlow {

struct Operator {
public:
  operator PCGOperatorAttrs() const;

public:
  PCGOperatorAttrs attrs;
  req<std::optional<std::string>> name;
};

FF_VISITABLE_STRUCT(Operator, attrs, name);

static_assert(is_well_behaved_value_type<Operator>::value);

} // namespace FlexFlow

#endif
