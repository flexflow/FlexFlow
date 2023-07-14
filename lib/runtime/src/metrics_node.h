#ifndef _FLEXFLOW_RUNTIME_SRC_METRICS_NODE_H
#define _FLEXFLOW_RUNTIME_SRC_METRICS_NODE_H

#include "metrics_functions.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct MetricsNode : public use_visitable_cmp<MetricsNode> {
public:
  MetricsNode() = delete;
  MetricsNode(Metrics const &,
              parallel_tensor_guid_t const &logit_tensor,
              parallel_tensor_guid_t const &label_tensor);

public:
  Metrics metrics;
  parallel_tensor_guid_t logit_tensor;
  parallel_tensor_guid_t label_tensor;
};

TaskInvocation compute_metrics(MetricsNode const &);
TaskInvocation update_metrics(MetricsNode const &);

} // namespace FlexFlow

#endif
