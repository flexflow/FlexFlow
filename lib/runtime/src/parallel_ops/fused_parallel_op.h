#ifndef _FLEXFLOW_FUSED_PARALLEL_OP_H
#define _FLEXFLOW_FUSED_PARALLEL_OP_H

#include "operator.h"
#include "parallel_op.h"

namespace FlexFlow {

class FusedParallelOp : public ParallelOp {
public:
  FusedParallelOp(FFModel &model,
                  const ParallelTensor input,
                  std::vector<ParallelOpInfo> const &parallel_ops);
  FusedParallelOp(FFModel &model, FusedParallelOpAttrs const &attrs, std::vector<ParallelTensor> const &inputs);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  bool append_parallel_op_info(
      std::vector<ParallelOpInfo> &parallel_ops) const override;
  void create_input_partition(FFModel &model) override;
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <typename T>
  static void
      forward_kernel(T const *input_ptr, T *output_ptr, size_t num_elements);
  template <typename T>
  static void backward_kernel(T const *output_grad_ptr,
                              T *input_grad_ptr,
                              size_t num_elements);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &mv,
                             CostMetrics &cost_metrics) const override;
  void set_parallel_ops(std::vector<ParallelOpInfo> const &_parallel_ops);
  bool check_no_redundant_parallel_ops(void) const;
public:
  stack_vector<ParallelOpInfo, MAX_NUM_FUSED_OPERATORS> parallel_ops;
};

}

#endif
