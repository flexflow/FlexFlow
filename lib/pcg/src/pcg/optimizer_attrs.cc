#include "pcg/optimizer_attrs.h"

namespace FlexFlow {

OptimizerAttrs next(OptimizerAttrs const &old_attrs) {
  if (old_attrs.has<AdamOptimizerAttrs>()) {
    AdamOptimizerAttrs old = old_attrs.get<AdamOptimizerAttrs>();
    double new_beta1_t = old.beta_t * old.beta1;
    double new_beta2_t = old.beta2_t * old.beta2;
    double new_alpha_t = old.alpha * sqrt(1 - new_beta2_t) / (1 - new_beta1_t);
    return OptimizerAttrs{AdamOptimizerAttrs{old.alpha,
                                             old.beta1,
                                             old.beta2,
                                             old.weight_decay,
                                             new_alpha_t,
                                             new_beta1_t,
                                             new_beta2_t,
                                             old.epsilon}};
  } else {
    return old_attrs;
  }
}

} // namespace FlexFlow
