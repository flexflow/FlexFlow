#include "compiler/cost_estimator/cost_metric.h"

namespace FlexFlow {

CostMetric zero_cost_metric() {
  return CostMetric{
      /*runtime=*/0,
      /*memory=*/0,
  };
}

CostMetric combine_cost_metrics_inter_device(CostMetric const &c1,
                                             CostMetric const &c2) {
  return CostMetric{c1.runtime + c2.runtime, c1.memory + c2.memory};
}

CostMetric
    combine_cost_metrics_inter_device(std::vector<CostMetric> const &costs) {
  CostMetric result = zero_cost_metric();
  for (CostMetric const &cost : costs) {
    result = combine_cost_metrics_inter_device(result, cost);
  }
  return result;
}

CostMetric combine_cost_metrics_intra_device_sequential(CostMetric const &c1,
                                                        CostMetric const &c2) {
  return CostMetric{c1.runtime + c2.runtime, std::max(c1.memory, c2.memory)};
}

CostMetric combine_cost_metrics_intra_device_sequential(
    std::vector<CostMetric> const &costs) {
  CostMetric result = zero_cost_metric();
  for (CostMetric const &cost : costs) {
    result = combine_cost_metrics_intra_device_sequential(result, cost);
  }
  return result;
}

CostMetric combine_cost_metrics_intra_device_parallel(CostMetric const &c1,
                                                      CostMetric const &c2) {
  return CostMetric{std::max(c1.runtime, c2.runtime),
                    std::max(c1.memory, c2.memory)};
}

CostMetric combine_cost_metrics_intra_device_parallel(
    std::vector<CostMetric> const &costs) {
  CostMetric result = zero_cost_metric();
  for (CostMetric const &cost : costs) {
    result = combine_cost_metrics_intra_device_parallel(result, cost);
  }
  return result;
}

} // namespace FlexFlow
