#ifndef _FLEXFLOW_CONCAT_H
#define _FLEXFLOW_CONCAT_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"

namespace FlexFlow {

struct ConcatParams {
  int axis;

  bool is_valid(std::vector<ParallelTensorShape> const &) const;
};

bool operator==(ConcatParams const &, ConcatParams const &);

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(FFHandler handle) : OpMeta(handle){};
  int legion_axis;
  char op_name[MAX_OPNAME];
};

class Concat : public Op {
public:
  using Params = ConcatParams;
  using Input = std::vector<ParallelTensor>;

  Concat(FFModel &model,
         int n,
         ParallelTensor const *inputs,
         int axis,
         char const *name);
  Concat(FFModel &model,
         ConcatParams const &params,
         std::vector<ParallelTensor> const &inputs,
         char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  bool get_int_parameter(PMParameter, int *) const override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  void init_meta(ConcatMeta *meta) const;
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void forward_kernel(float *output,
                             float const *const *inputs,
                             int num_inputs,
                             int axis,
                             Legion::Domain const &out_domain,
                             Legion::Domain const *in_domain,
                             ffStream_t stream);
  static void forward_kernel_wrapper(ConcatMeta const *m,
                                     float *output,
                                     float const *const *inputs,
                                     int num_inputs,
                                     int axis,
                                     Legion::Domain const &out_domain,
                                     Legion::Domain const *in_domain);
  static void backward_kernel(float const *output_grad,
                              float **input_grads,
                              int num_inputs,
                              int axis,
                              Legion::Domain const &out_grad_domain,
                              Legion::Domain const *in_grad_domain,
                              ffStream_t stream);
  static void backward_kernel_wrapper(ConcatMeta const *m,
                                      float const *output_grad,
                                      float **input_grads,
                                      int num_inputs,
                                      int axis,
                                      Legion::Domain const &out_grad_domain,
                                      Legion::Domain const *in_grad_domain);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  Params get_params() const;

public:
  int legion_axis;
};

}; // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ConcatParams> {
  size_t operator()(FlexFlow::ConcatParams const &) const;
};
}; // namespace std

#endif // _FLEXFLOW_CONCAT_H
