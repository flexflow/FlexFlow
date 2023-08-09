
#ifndef _FLEXFLOW_COMPILER_COST_ESTIMATE_H
#define _FLEXFLOW_COMPILER_COST_ESTIMATE_H

#include "op-attrs/operator_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/machine_view.h"

namespace FlexFlow {

struct ICostEstimator {
  virtual float estimate_cost(PCGOperatorAttrs const &op,
                              std::vector<ParallelTensorShape> const &inputs,
                              MachineView const &mv) const = 0;
  virtual float estimate_cost(ParallelTensorShape const &tensor_shape,
                              MachineView const &src,
                              MachineView const &dst) const = 0;

  ICostEstimator(ICostEstimator const &) = delete;
  ICostEstimator &operator=(ICostEstimator const &) = delete;

  virtual ~ICostEstimator();
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ICostEstimator);

struct CostEstimator {
  float estimate_cost(PCGOperatorAttrs const &op,
                      std::vector<ParallelTensorShape> const &inputs,
                      MachineView const &mv) const {
    return this->implementation_ptr->estimate_cost(op, inputs, mv);
  }

  float estimate_cost(ParallelTensorShape const &tensor_shape,
                      MachineView const &src,
                      MachineView const &dst) const {
    return this->implementation_ptr->estimate_cost(tensor_shape, src, dst);
  }

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<ICostEstimator, T>::value,
                                 CostEstimator>::type
      create(Args &&...args) {
    return CostEstimator(std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  std::shared_ptr<ICostEstimator> implementation_ptr;
};

} // namespace FlexFlow

#endif
