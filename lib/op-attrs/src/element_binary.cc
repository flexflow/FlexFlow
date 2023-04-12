#include "op-attrs/ops/element_binary.h"

namespace FlexFlow {

ElementBinaryAttrs::ElementBinaryAttrs(OperatorType _type, bool _should_broadcast_lhs, bool _should_broadcast_rhs)
  : type(_type), should_broadcast_lhs(_should_broadcast_lhs), should_broadcast_rhs(_should_broadcast_rhs) 
{ }

}
