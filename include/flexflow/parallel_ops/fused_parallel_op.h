#ifndef _FLEXFLOW_FUSED_PARALLEL_OP_H
#define _FLEXFLOW_FUSED_PARALLEL_OP_H

#include "parallel_op.h"

namespace FlexFlow {

class FusedParallelOp : public ParallelOp {
public:
  FusedParallelOp(FFModel& model,
                  const ParallelTensor input,
                  const std::vector<ParallelOpInfo>& parallel_ops);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void reset_idx(const FFModel&) override {assert(0);}
  void pipeinit(const FFModel&)  override {assert(0);}
  void pipeforward(const FFModel&)  override {assert(0);}
  void pipebackward(const FFModel&)  override {assert(0);}
  bool append_parallel_op_info(std::vector<ParallelOpInfo>& parallel_ops) const override;
  void create_input_partition(FFModel& model) override;
  static void forward_task(
      const Legion::Task *task,
      const std::vector<Legion::PhysicalRegion> &regions,
      Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(
      const Legion::Task *task,
      const std::vector<Legion::PhysicalRegion> &regions,
      Legion::Context ctx, Legion::Runtime *runtime);
  template<typename T>
  static void forward_kernel(
      const T* input_ptr,
      T* output_ptr,
      size_t num_elements);
  template<typename T>
  static void backward_kernel(
      const T* output_grad_ptr,
      T* input_grad_ptr,
      size_t num_elements);
  bool measure_operator_cost(
      Simulator* sim,
      const MachineView& mv,
      CostMetrics& cost_metrics) const override;
  void set_parallel_ops(const std::vector<ParallelOpInfo>& _parallel_ops);
  bool check_no_redundant_parallel_ops(void) const;

  size_t get_params_hash() const override;
public:
  int num_parallel_ops;
  ParallelOpInfo parallel_ops[MAX_NUM_FUSED_OPERATORS];
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_FUSED_PARALLEL_OP_H
