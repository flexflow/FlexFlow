#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_COST_METRIC_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_COST_METRIC_H

#include "compiler/cost_estimator/cost_metric.dtg.h"
#include <vector>

namespace FlexFlow {

CostMetric zero_cost_metric();

CostMetric combine_cost_metrics_inter_device(CostMetric const &c1,
                                             CostMetric const &c2);
CostMetric
    combine_cost_metrics_inter_device(std::vector<CostMetric> const &costs);

CostMetric combine_cost_metrics_intra_device_sequential(CostMetric const &c1,
                                                        CostMetric const &c2);
CostMetric combine_cost_metrics_intra_device_sequential(
    std::vector<CostMetric> const &costs);

CostMetric combine_cost_metrics_intra_device_parallel(CostMetric const &c1,
                                                      CostMetric const &c2);
CostMetric combine_cost_metrics_intra_device_parallel(
    std::vector<CostMetric> const &costs);

} // namespace FlexFlow

#endif
