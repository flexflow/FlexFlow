#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_METRICS_KERNELS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_METRICS_KERNELS_H

#include "perf_metrics.h"

namespace FlexFlow {

void update_metrics_sparse_label_kernel(ffStream_t,
                                        Metrics const &,
                                        float const *logit_ptr,
                                        int const *label_ptr,
                                        int num_samples,
                                        int num_classes,
                                        PerfMetrics &perf_zc);
void update_metrics_label_kernel(ffStream_t,
                                 Metrics const &,
                                 float const *logit_ptr,
                                 float const *label_ptr,
                                 int num_samples,
                                 int num_classes,
                                 PerfMetrics &perf_zc);

} // namespace FlexFlow

#endif
