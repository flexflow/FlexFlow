#ifndef _FLEXFLOW_NOOP_H
#define _FLEXFLOW_NOOP_H

#include "flexflow/model.h"

namespace FlexFlow {

class NoOp : public Op {
public:
  NoOp(FFModel& model,
       OperatorType type,
       const ParallelTensor output,
       const char* name = NULL);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  
  size_t get_params_hash() const override;
  tl::optional<RecordFormatter> as_dot() const override;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_NOOP_H
