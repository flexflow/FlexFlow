#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "flexflow/ops/concat.h"
#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/element_binary.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/linear.h"
#include "flexflow/ops/pool_2d_params.h"
#include "flexflow/ops/dropout.h"
#include "flexflow/ops/cast.h"
#include "flexflow/ops/embedding.h"
#include "mpark/variant.hpp"

namespace mp = mpark;

namespace FlexFlow {

using OperatorParameters = mp::variant<Conv2DParams,
                                       LinearParams,
                                       ConcatParams,
                                       ElementBinaryParams,
                                       ElementUnaryParams,
                                       Pool2DParams,
                                       CastParams,
                                       DropoutParams,
                                       EmbeddingParams>;

}; // namespace FlexFlow

#endif // _OPERATOR_PARAMS_H
