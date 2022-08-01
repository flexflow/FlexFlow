#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "mpark/variant.hpp"
#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/linear.h"
#include "flexflow/ops/concat.h"
#include "flexflow/ops/element_binary.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/pool_2d_params.h"

namespace mp = mpark;

namespace FlexFlow {

using OperatorParameters = mp::variant<
Conv2DParams,
LinearParams,
ConcatParams,
ElementBinaryParams,
ElementUnaryParams,
Pool2DParams
>;

}; // namespace FlexFlow

#endif // _OPERATOR_PARAMS_H
