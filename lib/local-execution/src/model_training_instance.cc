#include "local-execution/model_training_instance.h"

namespace FlexFlow {

ModelTrainingInstance next(ModelTrainingInstance const &old_training_instance) {
  if (old_training_instance.optimizer_attrs.has<AdamOptimizerAttrs>()) {
    AdamOptimizerAttrs old =
        old_training_instance.optimizer_attrs.get<AdamOptimizerAttrs>();
    double new_beta1_t = old.beta_t * old.beta1;
    double new_beta2_t = old.beta2_t * old.beta2;
    double new_alpha_t = old.alpha * sqrt(1 - new_beta2_t) / (1 - new_beta1_t);
    OptimizerAttrs new_attrs =
        OptimizerAttrs{AdamOptimizerAttrs{old.alpha,
                                          old.beta1,
                                          old.beta2,
                                          old.weight_decay,
                                          new_alpha_t,
                                          new_beta1_t,
                                          new_beta2_t,
                                          old.epsilon}};
    return ModelTrainingInstance{old_training_instance.loss_attrs,
                                 old_training_instance.label_tensor,
                                 old_training_instance.logit_tensor,
                                 new_attrs};
  }
  return old_training_instance;
}

} // namespace FlexFlow
