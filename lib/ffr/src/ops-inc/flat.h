#ifndef _FLEXFLOW_FLAT_H
#define _FLEXFLOW_FLAT_H

#include "device.h"
#include "fftype.h"
#include "layer.h"
#include "flexflow/node.h"
#include "op_meta.h"
#include "operator.h"
#include "op-meta/flat_params.h"

namespace FlexFlow {

class FlatMeta : public OpMeta {
public:
  FlatMeta(FFHandler handle) : OpMeta(handle){};
};

class Flat : public Op {
public:
  using Params = FlatParams;
  using Input = ParallelTensor;

  Flat(FFModel &model, const ParallelTensor input, char const *name);
  Flat(FFModel &model,
       Params const &params,
       const Input input,
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
  static void forward_kernel(float const *input_ptr,
                             float *output_ptr,
                             size_t num_elements,
                             ffStream_t stream);
  static void forward_kernel_wrapper(float const *input_ptr,
                                     float *output_ptr,
                                     size_t num_elements);
  static void backward_kernel(float *input_grad_ptr,
                              float const *output_grad_ptr,
                              size_t num_elements,
                              ffStream_t stream);
  static void backward_kernel_wrapper(float *input_grad_ptr,
                                      float const *output_grad_ptr,
                                      size_t num_elements);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  Legion::Domain get_input_tensor_shape(ParallelConfig const &pc,
                                        int input_idx,
                                        int part_idx) const override;

  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  static void
      construct_output_mappings(std::vector<ParallelDimMappingRecord> &);

  Params get_params() const;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_FLAT_H
