#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_OPERATOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_OPERATOR_H

#include "op-attrs/operator_attrs.h"
#include "utils/optional.h"
#include "utils/stack_string.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct Operator {
public:
  operator PCGOperatorAttrs() const;

public:
  req<PCGOperatorAttrs> attrs;
  req<optional<stack_string<MAX_OPNAME>>> name;
};

FF_VISITABLE_STRUCT(Operator, attrs, name);

} // namespace FlexFlow

#endif
