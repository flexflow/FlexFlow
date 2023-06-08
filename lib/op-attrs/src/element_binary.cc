#include "op-attrs/ops/element_binary.h"

namespace FlexFlow {

ElementBinaryAttrs::ElementBinaryAttrs(OperatorType _type,
                                       DataType _compute_type,
                                       bool _should_broadcast_lhs,
                                       bool _should_broadcast_rhs)
    : type(_type), compute_type(_compute_type),
      should_broadcast_lhs(_should_broadcast_lhs),
      should_broadcast_rhs(_should_broadcast_rhs) {}

} // namespace FlexFlow
