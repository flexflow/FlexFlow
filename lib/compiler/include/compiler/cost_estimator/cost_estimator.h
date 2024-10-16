#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_COST_ESTIMATOR_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_COST_ESTIMATOR_H

#include "compiler/cost_estimator/cost_metric.dtg.h"
#include "compiler/cost_estimator/op_cost_estimate_key.dtg.h"
#include "compiler/cost_estimator/tensor_set_movement.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "pcg/machine_view.dtg.h"
#include <vector>

namespace FlexFlow {

struct ICostEstimator {
  virtual CostMetric estimate_cost(OpCostEstimateKey const &) const = 0;
  virtual CostMetric estimate_cost(TensorSetMovement const &) const = 0;

  ICostEstimator() = default;
  ICostEstimator(ICostEstimator const &) = delete;
  ICostEstimator &operator=(ICostEstimator const &) = delete;

  virtual ~ICostEstimator() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ICostEstimator);

struct CostEstimator {
  CostMetric estimate_cost(OpCostEstimateKey const &k) const;
  CostMetric estimate_cost(TensorSetMovement const &m) const;

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<ICostEstimator, T>::value,
                                 CostEstimator>::type
      create(Args &&...args) {
    return CostEstimator(std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  CostEstimator(std::shared_ptr<ICostEstimator> implementation_ptr);

private:
  std::shared_ptr<ICostEstimator> implementation_ptr;
};

} // namespace FlexFlow

#endif
