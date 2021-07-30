#ifndef _FLEXFLOW_FUSED_PARALLEL_OP_H
#define _FLEXFLOW_FUSED_PARALLEL_OP_H

#include "parallel_op.h"

namespace FlexFlow {

class FusedParallelOp : public ParallelOp {
public:
  FusedParallelOp(FFModel& model,
                  const Tensor input,
                  const std::vector<ParallelOpInfo>& parallel_ops);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  bool append_parallel_op_info(std::vector<ParallelOpInfo>& parallel_ops) const;
  void create_input_partition(FFModel& model);
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
      const ParallelConfig& pc,
      CostMetrics& cost_metrics) const;
  void set_parallel_ops(const std::vector<ParallelOpInfo>& _parallel_ops);
  bool check_no_redundant_parallel_ops(void) const;

  size_t get_params_hash() const override;
public:
  int num_parallel_ops;
  ParallelOpInfo parallel_ops[MAX_NUM_FUSED_OPERATORS];
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_FUSED_PARALLEL_OP_H
