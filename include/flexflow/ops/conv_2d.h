#ifndef _FLEXFLOW_CONV_2D_H
#define _FLEXFLOW_CONV_2D_H

#include "flexflow/node.h"
#include "flexflow/operator.h"
#include "flexflow/ops/conv_2d_params.h"

namespace FlexFlow {

class FFModel;
class Layer;

namespace Conv2DInput {
static constexpr int INDEX = 0;

enum { WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3, REPLICA = 4, NUMDIM };
} // namespace Conv2DInput

namespace Conv2DOutput {
enum { WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3, REPLICA = 4, NUMDIM };
}

namespace Conv2DKernel {
static constexpr int INDEX = 0;

enum {
  WIDTH = 0,
  HEIGHT = 1,
  CHANNEL_IN = 2,
  CHANNEL_OUT = 3,
  REPLICA = 4,
  NUMDIM
};
} // namespace Conv2DKernel

namespace Conv2DBias {
static constexpr int INDEX = 1;

enum {
  CHANNEL = 0,
  REPLICA_1 = 1,
  REPLICA_2 = 2,
  REPLICA_3 = 3,
  REPLICA_4 = 4,
  NUMDIM
};
} // namespace Conv2DBias

class Conv2D : public Op {
public:
  using Params = Conv2DParams;
  using Input = ParallelTensor;

  Conv2D(FFModel &model,
         LayerID const &layer_guid,
         const ParallelTensor input,
         int outChannels,
         int kernelH,
         int kernelW,
         int strideH,
         int strideW,
         int paddingH,
         int paddingW,
         ActiMode activation,
         int groups,
         bool use_bias,
         bool allocate_weights,
         char const *name);
  Conv2D(FFModel &model,
         Conv2D const &other,
         const ParallelTensor input,
         bool allocate_weights);
  Conv2D(FFModel &model,
         Params const &params,
         Input const input,
         char const *name = nullptr,
         bool allocate_weights = false);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  // void update(const FFModel&);
  void print_layer(FFModel const &model) override;
  // Parameter* get_parameter(int index);
  // void create_weights(FFModel& model);
  // void create_input_partition(FFModel& model);
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
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  bool estimate_sync_cost(Simulator *sim,
                          MachineView const &pc,
                          CostMetrics &cost_metrics) const override;

  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);

  static void
      construct_output_mappings(std::vector<ParallelDimMappingRecord> &);
  static void construct_mappings(std::vector<ParallelDimMappingRecord> &,
                                 bool use_bias);
  static void construct_weight_mappings(std::vector<ParallelDimMappingRecord> &,
                                        bool use_bias);

  Params get_params() const;

  tl::optional<RecordFormatter> as_dot() const override;

public:
  int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
      padding_h, padding_w;
  ActiMode activation;
  int groups;
  bool use_bias;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_CONV_2D_H
