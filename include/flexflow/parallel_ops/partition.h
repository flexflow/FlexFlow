#ifndef _FLEXFLOW_PARTITION_H
#define _FLEXFLOW_PARTITION_H

#include "parallel_op.h"

namespace FlexFlow {

class RepartitionMeta : public OpMeta {
public:
  RepartitionMeta(FFHandler handle);
  DataType data_type;
};

class Repartition : public ParallelOp {
public:
  Repartition(FFModel &model,
              const ParallelTensor input,
              int repartition_legion_dim,
              int repartition_degree,
              char const *name = NULL);
  void create_input_partition(FFModel &model) override;
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  bool get_int_parameter(PMParameter, int *) const override;
  bool append_parallel_op_info(
      std::vector<ParallelOpInfo> &parallel_ops) const override;
  static OpMeta *init_task(Legion::Task const *task,
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
  template <typename T>
  static void
  forward_task_with_type(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  template <typename T>
  static void
  backward_task_with_type(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
  template <typename T>
  static void
  forward_kernel(const T *input_ptr, T *output_ptr, size_t num_elements);
  template <typename T>
  static void backward_kernel(const T *output_grad_ptr,
                              T *input_grad_ptr,
                              size_t num_elements);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  size_t get_params_hash() const override;
  tl::optional<RecordFormatter> as_dot() const override;

public:
  int repartition_dim, repartition_degree;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_PARTITION_H
