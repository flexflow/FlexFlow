#pragma once

#include "flexflow/model.h"
#include "flexflow/ops/gather_params.h"

namespace FlexFlow {

class Gather;

class GatherMeta : public OpMeta {
public:
  GatherMeta(FFHandler handler, Gather const *gather);

public:
  int legion_dim;
};

class Gather : public Op {
public:
  using Params = GatherParams;
  using Input = std::pair<ParallelTensor, ParallelTensor>;
  Gather(FFModel &model,
         Params const &params,
         Input const &input,
         char const *name = nullptr);
  Gather(FFModel &model,
         const ParallelTensor input,
         const ParallelTensor index,
         int legion_dim,
         char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
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
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <typename TI>
  static void forward_kernel(float const *input_ptr,
                             TI const *index_ptr,
                             float *output_ptr,
                             Legion::coord_t output_size,
                             Legion::coord_t stride,
                             Legion::coord_t dim_size,
                             ffStream_t stream);
  static void forward_kernel_wrapper(GatherMeta const *m,
                                     GenericTensorAccessorR const &input,
                                     GenericTensorAccessorR const &index,
                                     GenericTensorAccessorW const &output);
  template <typename TI>
  static void backward_kernel(float const *output_grad_ptr,
                              TI const *index_ptr,
                              float *input_grad_ptr,
                              Legion::coord_t output_size,
                              Legion::coord_t stride,
                              Legion::coord_t dim_size,
                              ffStream_t stream);
  static void backward_kernel_wrapper(GatherMeta const *m,
                                      GenericTensorAccessorR const &output_grad,
                                      GenericTensorAccessorR const &index,
                                      GenericTensorAccessorW const &input_grad);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  Params get_params() const;

public:
  int legion_dim;
};

}; // namespace FlexFlow
