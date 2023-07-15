#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_OPTIMIZER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_OPTIMIZER_H

#include "utils/variant.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct SGDOptimizer {
  req<double> lr;
  req<double> momentum;
  req<bool> nesterov;
  req<double> weight_decay;
};
FF_VISITABLE_STRUCT(SGDOptimizer, lr, momentum, nesterov, weight_decay);

struct AdamOptimizer {
  req<double> alpha;
  req<double> beta1;
  req<double> beta2;
  req<double> weight_decay;
  req<double> epsilon;
  req<double> alpha_t;
  req<double> beta_t;
  req<double> beta2_t;
};
FF_VISITABLE_STRUCT(AdamOptimizer,
                    alpha,
                    beta1,
                    beta2,
                    weight_decay,
                    epsilon,
                    alpha_t,
                    beta_t,
                    beta2_t);

using Optimizer = variant<SGDOptimizer, AdamOptimizer>;

} // namespace FlexFlow

#endif
