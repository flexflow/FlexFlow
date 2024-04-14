#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_OPTIMIZER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_OPTIMIZER_H

#include "utils/variant.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct SGDOptimizer {
  double lr;
  double momentum;
  bool nesterov;
  req<double> weight_decay;
};
FF_VISITABLE_STRUCT(SGDOptimizer, lr, momentum, nesterov, weight_decay);

struct AdamOptimizer {
  double alpha;
  double beta1;
  double beta2;
  double weight_decay;
  double epsilon;
  double alpha_t;
  double beta_t;
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

using Optimizer = std::variant<SGDOptimizer, AdamOptimizer>;

} // namespace FlexFlow

#endif
