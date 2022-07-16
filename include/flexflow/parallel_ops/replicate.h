#ifndef _FLEXFLOW_REPLICATE_H
#define _FLEXFLOW_REPLICATE_H

#include "parallel_op.h"

namespace FlexFlow {

class Replicate : public ParallelOp {
public:
  Replicate(FFModel& model,
      const ParallelTensor input,
      int replicate_legion_dim,
      int replicate_degree,
      const char* name = NULL);
  void create_input_partition(FFModel& model) override;
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void pipeinit(const FFModel&)  override {assert(0);}
  void pipeforward(const FFModel&)  override {assert(0);}
  void pipebackward(const FFModel&)  override {assert(0);}
  bool get_int_parameter(PMParameter, int*) const override;
  bool append_parallel_op_info(std::vector<ParallelOpInfo>& parallel_ops) const override;
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
      size_t num_elements,
      size_t num_replicas);
  bool measure_operator_cost(
      Simulator* sim,
      const MachineView& pc,
      CostMetrics& cost_metrics) const override;

  size_t get_params_hash() const override;
public:
  int replicate_dim, replicate_degree;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_REPLICATE_H
