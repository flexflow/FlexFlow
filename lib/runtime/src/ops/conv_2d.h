#ifndef _FLEXFLOW_CONV_2D_H
#define _FLEXFLOW_CONV_2D_H

#include "layer_id.h"
#include "operator.h"
#include "layer.h"

namespace FlexFlow {

namespace Conv2DInput {
static constexpr int INDEX = 0;

enum { WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3, REPLICA = 4, NUMDIM };
} 

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
} 

/* namespace Conv2DBias { */
/* static constexpr int INDEX = 1; */

class Conv2D : public Op {
public:
  using Attrs = Conv2DAttrs;

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
         Attrs const &attrs,
         char const *name = nullptr,
         bool allocate_weights = false);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  // void update(const FFModel&);
  // Parameter* get_parameter(int index);
  // void create_weights(FFModel& model);
  // void create_input_partition(FFModel& model);
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);

  static PerDeviceOpState *init_task(Legion::Task const *task,
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

  /* static void */
  /*     construct_output_mappings(std::vector<ParallelDimMappingRecord> &); */
  /* static void construct_mappings(std::vector<ParallelDimMappingRecord> &, */
  /*                                bool use_bias); */
  /* static void construct_weight_mappings(std::vector<ParallelDimMappingRecord> &, */
  /*                                       bool use_bias); */

  tl::optional<RecordFormatter> as_dot() const override;

public:
  int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
      padding_h, padding_w;
  ActiMode activation;
  int groups;
  bool use_bias;
};

}

#endif 
