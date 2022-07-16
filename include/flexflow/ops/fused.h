#ifndef _FLEXFLOW_FUSED_H_
#define _FLEXFLOW_FUSED_H_

#include "flexflow/model.h"

namespace FlexFlow {

class FusedOp;
class FusedOpMeta {
public:
  FusedOpMeta(void) {}
  OpMeta* meta[MAX_NUM_FUSED_OPERATORS];
  FusedOp* fused_op;
  int numOperators;
};

class FusedOp : public Op {
public:
  enum SourceType {
    SOURCE_NONE,
    SOURCE_INPUT,
    SOURCE_WEIGHT,
    SOURCE_OUTPUT,
  };
  FusedOp(FFModel& model,
          Op* op);
  bool add_operator(FFModel& model, Op* op);
  ParallelTensor init_inout(FFModel& model, const ParallelTensor input) {assert(0); return ParallelTensor();}
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void pipeinit(const FFModel&)  override {assert(0);}
  void pipeforward(const FFModel&)  override {assert(0);}
  void pipebackward(const FFModel&)  override {assert(0);}
  void print_layer(const FFModel& model) override {assert(0);}
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator* sim,
                             const MachineView& pc,
                             CostMetrics& cost_metrics) const override;
public:
  FFIterationConfig iter_config;
  int op_num_inputs[MAX_NUM_FUSED_OPERATORS];
  int op_num_weights[MAX_NUM_FUSED_OPERATORS];
  int op_num_outputs[MAX_NUM_FUSED_OPERATORS];
  OperatorType op_op_type[MAX_NUM_FUSED_OPERATORS];
  SourceType op_input_source[MAX_NUM_FUSED_TENSORS];
  SourceType op_weight_source[MAX_NUM_FUSED_TENSORS];
  SourceType op_output_source[MAX_NUM_FUSED_TENSORS];
  int op_input_idx[MAX_NUM_FUSED_TENSORS];
  int op_weight_idx[MAX_NUM_FUSED_TENSORS];
  int op_output_idx[MAX_NUM_FUSED_TENSORS];
  Op* operators[MAX_NUM_FUSED_OPERATORS];
  FusedOpMeta fused_meta[MAX_NUM_WORKERS];
  int numOperators;
};

}; // namespace FlexFlow

#endif
