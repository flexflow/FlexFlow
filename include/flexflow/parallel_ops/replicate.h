#ifndef _FLEXFLOW_REPLICATE_H
#define _FLEXFLOW_REPLICATE_H

#include "parallel_op.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"

namespace FlexFlow {

struct ReplicateParams {
  int replicate_legion_dim;
  int replicate_degree;
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(ReplicateParams const &, ReplicateParams const &);

class Replicate : public ParallelOp {
public:
  using Params = ReplicateParams;
  using Input = ParallelTensor;

  Replicate(FFModel &model,
            const ParallelTensor input,
            int replicate_legion_dim,
            int replicate_degree,
            char const *name = NULL);
  Replicate(FFModel &model,
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
  static void
      forward_kernel(const T *input_ptr, T *output_ptr, size_t num_elements);
  template <typename T>
  static void backward_kernel(const T *output_grad_ptr,
                              T *input_grad_ptr,
                              size_t num_elements,
                              size_t num_replicas);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  Params get_params() const;

public:
  int replicate_dim, replicate_degree;
};

}; // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ReplicateParams> {
  size_t operator()(FlexFlow::ReplicateParams const &) const;
}
} // namespace std

#endif // _FLEXFLOW_REPLICATE_H
