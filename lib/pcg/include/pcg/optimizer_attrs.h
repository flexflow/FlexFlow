#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPTIMIZER_ATTRS_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPTIMIZER_ATTRS_H

#include "pcg/optimizer_attrs.dtg.h"

namespace FlexFlow {
  
OptimizerAttrs make_empty_sgd_attrs();
OptimizerAttrs make_empty_adam_attrs();

} // namespace FlexFlow

#endif
