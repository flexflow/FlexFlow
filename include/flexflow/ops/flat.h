#ifndef _FLEXFLOW_FLAT_H
#define _FLEXFLOW_FLAT_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/params/flat_params.h"

namespace FlexFlow {

namespace FlatInput {
constexpr int NUMDIM = 5, WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3,
              REPLICA = 4;
}

namespace FlatOutput {
constexpr int NUMDIM = 3, CHANNEL = 0, SAMPLE = 1, REPLICA = 2;
}

class FlatMeta : public OpMeta {
public:
  FlatMeta(FFHandler handle) : OpMeta(handle){};
};

class Flat : public Op {
public:
  using Params = FlatParams;

  Flat(FFModel &model, const ParallelTensor input, char const *name);
  Flat(FFModel &model,
       Params const &params,
       std::vector<ParallelTensor> const &input,
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
                               std::vector<ParallelTensor> const &);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  static void
      construct_output_mappings(std::vector<ParallelDimMappingRecord> &);

  Params get_params() const;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_FLAT_H
