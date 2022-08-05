#ifndef _FLEXFLOW_REDUCTION_H
#define _FLEXFLOW_REDUCTION_H

#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/node.h"
#include "flexflow/device.h"
#include "flexflow/layer.h"

namespace FlexFlow {

struct ReductionParams {
  int reduction_legion_dim;
  int reduction_degree;
  bool is_valid(const ParallelTensorShape &) const;
};
bool operator==(const ReductionParams &, const ReductionParams &);

class Reduction : public ParallelOp {
public:
  using Params = ReductionParams;
  using Input = ParallelTensor;

  Reduction(FFModel &model,
            const ParallelTensor input,
            int reduction_legion_dim,
            int reduction_degree,
            char const *name = NULL);
  Reduction(FFModel &model,
            Params const &params,
            const Input input,
            char const *name = nullptr);
  void create_input_partition(FFModel &model) override;
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  bool get_int_parameter(PMParameter, int *) const override;
  bool append_parallel_op_info(
      std::vector<ParallelOpInfo> &parallel_ops) const override;
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <typename T>
  static void forward_kernel(const T *input_ptr,
                             T *output_ptr,
                             size_t num_elements,
                             size_t num_replicas);
  template <typename T>
  static void backward_kernel(const T *output_grad_ptr,
                              T *input_grad_ptr,
                              size_t num_elements);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  Params get_params() const;

public:
  int reduction_dim, reduction_degree;
};

}; // namespace FlexFlow

namespace std {
  template <>
  struct hash<FlexFlow::ReductionParams> {
    size_t operator()(const FlexFlow::ReductionParams&) const;
  }
} // namespace std

#endif // _FLEXFLOW_REDUCTION_H
