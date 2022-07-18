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
  NoOp(FFModel& model,
       OperatorType type,
       size_t input_tensor_guid,
       const ParallelTensor output,
       const char* name = NULL);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void pipeinit(const FFModel&)  override;
  void pipeforward(const FFModel&)  override;
  void pipebackward(const FFModel&)  override;
  void print_layer(const FFModel& model) override {assert(0);}
  bool measure_operator_cost(Simulator* sim,
                             const MachineView& pc,
                             CostMetrics& cost_metrics) const override;
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  
  size_t get_params_hash() const override;
  tl::optional<RecordFormatter> as_dot() const override;
public:
  size_t input_tensor_guid;
  int init_output_idx = 0;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_NOOP_H
