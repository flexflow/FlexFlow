#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_OPTIMIZER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_OPTIMIZER_H

#include "pcg/optimizers/adam_optimizer_attrs.h"
#include "pcg/optimizers/sgd_optimizer_attrs.h"
#include <variant>

namespace FlexFlow {

using OptimizerAttrs = std::variant<SGDOptimizerAttrs, AdamOptimizerAttrs>;

} // namespace FlexFlow

#endif
