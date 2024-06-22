
#ifndef _FLEXFLOW_LOCAL_EXECUTION_COST_ESTIMATE_H
#define _FLEXFLOW_LOCAL_EXECUTION_COST_ESTIMATE_H

#include "local-execution/cost_details.dtg.h"
#include "local-execution/local_training_backing.h"
#include "op-attrs/operator_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"

namespace FlexFlow {

struct ICostEstimator {
  virtual CostDetails
      estimate_cost(PCGOperatorAttrs const &op,
                    std::vector<ParallelTensorShape> const &inputs,
                    std::vector<ParallelTensorAttrs> const &weights,
                    std::vector<ParallelTensorAttrs> const &outputs,
                    MachineView const &mv) const = 0;
  virtual float estimate_cost(ParallelTensorShape const &tensor_shape,
                              MachineView const &src,
                              MachineView const &dst) const = 0;

  ICostEstimator() = default;
  ICostEstimator(ICostEstimator const &) = delete;
  ICostEstimator &operator=(ICostEstimator const &) = delete;

  virtual ~ICostEstimator() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ICostEstimator);

struct CostEstimator {
  CostDetails estimate_cost(PCGOperatorAttrs const &op,
                            std::vector<ParallelTensorShape> const &inputs,
                            std::vector<ParallelTensorAttrs> const &weights,
                            std::vector<ParallelTensorAttrs> const &outputs,
                            MachineView const &mv) const {
    return this->implementation_ptr->estimate_cost(
        op, inputs, weights, outputs, mv);
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
  CostEstimator(std::shared_ptr<ICostEstimator> implementation_ptr)
      : implementation_ptr(implementation_ptr) {}
  std::shared_ptr<ICostEstimator> implementation_ptr;
};

} // namespace FlexFlow

#endif
