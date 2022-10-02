#ifndef _FLEXFLOW_CONCAT_H
#define _FLEXFLOW_CONCAT_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/concat_params.h"
#include "flexflow/accessor.h"

namespace FlexFlow {

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
  static void forward_kernel(GenericTensorAccessorW const &output,
                             GenericTensorAccessorR const *inputs,
                             int num_inputs,
                             int axis,
                             ffStream_t stream);
  static void forward_kernel_wrapper(ConcatMeta const *m,
                                     GenericTensorAccessorW const &output,
                                     GenericTensorAccessorR const *inputs,
                                     int num_inputs,
                                     int axis);
  static void backward_kernel(GenericTensorAccessorR const &output_grad,
                              GenericTensorAccessorW const *input_grads,
                              int num_inputs,
                              int axis,
                              ffStream_t stream);
  static void backward_kernel_wrapper(ConcatMeta const *m,
                                      GenericTensorAccessorR const &output_grad,
                                      GenericTensorAccessorW const *input_grads,
                                      int num_inputs,
                                      int axis);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  Params get_params() const;

public:
  int legion_axis;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_CONCAT_H
