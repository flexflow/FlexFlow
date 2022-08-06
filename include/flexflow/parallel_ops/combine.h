#ifndef _FLEXFLOW_COMBINE_H
#define _FLEXFLOW_COMBINE_H

#include "parallel_op.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"

namespace FlexFlow {

struct CombineParams {
  int combine_legion_dim;
  int combine_degree;
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(CombineParams const &, CombineParams const &);

class CombineMeta : public OpMeta {
public:
  CombineMeta(FFHandler handle);
  DataType data_type;
};

class Combine : public ParallelOp {
public:
  using Params = RepartitionParams;
  using Input = ParallelTensor;

  Combine(FFModel &model,
          const ParallelTensor input,
          int combine_legion_dim,
          int combine_degree,
          char const *name = NULL);
  Combine(FFModel &model,
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
  static void backward_task_with_type(
      Legion::Task const *task,
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
                             MachineView const &mv,
                             CostMetrics &cost_metrics) const override;

  Params get_params() const;
  tl::optional<RecordFormatter> as_dot() const override;

public:
  int combine_dim, combine_degree;
};

}; // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::CombineParams> {
  size_t operator()(FlexFlow::CombineParams const &) const;
}
} // namespace std

#endif // _FLEXFLOW_COMBINE_H
