#include "pcg/optimizer_attrs.h"

namespace FlexFlow {
  
OptimizerAttrs make_empty_sgd_attrs() {
  return OptimizerAttrs{SGDOptimizerAttrs{0.0, 0.0, false, 0.0}};
}

OptimizerAttrs make_empty_adam_attrs() {
  return OptimizerAttrs{AdamOptimizerAttrs{0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0}};
}

} // namespace FlexFlow
