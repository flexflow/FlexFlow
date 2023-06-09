#include "op-attrs/ops/element_unary.h"

namespace FlexFlow {

ElementScalarUnaryAttrs::ElementScalarUnaryAttrs(OperatorType _op,
                                                 float _scalar)
    : op(_op), scalar(_scalar) {}

ElementUnaryAttrs::ElementUnaryAttrs(OperatorType _op) : op(_op) {}

} // namespace FlexFlow
