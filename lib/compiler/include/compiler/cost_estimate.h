
#ifndef _FLEXFLOW_COMPILER_COST_ESTIMATE_H
#define _FLEXFLOW_COMPILER_COST_ESTIMATE_H

#include "op-attrs/operator_attrs.h"
#include "pcg/machine_view.h"

namespace FlexFlow {

struct ICostEstimator {
  virtual float estimate_cost(PCGOperatorAttrs const &op,
                              std::vector<ParallelTensorShape> const &inputs,
                              MachineView const &mv) const = 0;
  virtual float estimate_cost(ParallelTensorShape const &tensor_shape,
                              MachineView const &src,
                              MachineView const &dst) const = 0;
};

}

#endif