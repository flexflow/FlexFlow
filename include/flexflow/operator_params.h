#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "mpark/variant.hpp"
#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/linear.h"

namespace mp = mpark;

namespace FlexFlow {

using OperatorParameters = mp::variant<
Conv2DParams,
LinearParams
>;

}; // namespace FlexFlow

#endif // _OPERATOR_PARAMS_H