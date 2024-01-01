#ifndef _FLEXFLOW_FUSED_H_
#define _FLEXFLOW_FUSED_H_

#include "flexflow/model.h"

namespace FlexFlow {

class FusedOp;
class FusedOpMeta {
public:
  FusedOpMeta(void) {}
  OpMeta *meta[MAX_NUM_FUSED_OPERATORS];
  FusedOp *fused_op;
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
  FusedOp(FFModel &model, Op *op);
  static bool use_same_regions(
      ParallelTensor const source_tensor,
      ParallelTensor const target_tensor,
      std::unordered_map<ParallelTensor, std::vector<ParallelTensor>>
          *pt_mapping = nullptr);
  bool add_operator(
      FFModel &model,
      Op *op,
      std::unordered_map<ParallelTensor, std::vector<ParallelTensor>>
          *parallel_tensor_mapping = nullptr);
  ParallelTensor init_inout(FFModel &model, const ParallelTensor input) {
    assert(0);
    return ParallelTensor();
  }
  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

public:
  FFIterationConfig iter_config;
  int op_num_inputs[MAX_NUM_FUSED_OPERATORS];
  int op_num_weights[MAX_NUM_FUSED_OPERATORS];
  int op_num_outputs[MAX_NUM_FUSED_OPERATORS];
  OperatorType op_op_type[MAX_NUM_FUSED_OPERATORS];
  SourceType op_input_source[MAX_NUM_FUSED_TENSORS];
  SourceType op_weight_source[MAX_NUM_FUSED_TENSORS];
  SourceType op_output_source[MAX_NUM_FUSED_TENSORS];
  DataType input_data_types[MAX_NUM_INPUTS];
  DataType weight_data_types[MAX_NUM_WEIGHTS];
  DataType output_data_types[MAX_NUM_OUTPUTS];
  int op_input_idx[MAX_NUM_FUSED_TENSORS];
  int op_weight_idx[MAX_NUM_FUSED_TENSORS];
  int op_output_idx[MAX_NUM_FUSED_TENSORS];
  Op *operators[MAX_NUM_FUSED_OPERATORS];
  FusedOpMeta fused_meta[MAX_NUM_WORKERS];
  int numOperators;
};

}; // namespace FlexFlow

#endif
