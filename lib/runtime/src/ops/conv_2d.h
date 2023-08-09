#ifndef _FLEXFLOW_CONV_2D_H
#define _FLEXFLOW_CONV_2D_H

#include "op-attrs/ops/conv_2d.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<CONV2D_INIT_TASK_ID>();
template <>
void register_task<CONV2D_FWD_TASK_ID>();
template <>
void register_task<CONV2D_BWD_TASK_ID>();

OpTaskInvocation init(Conv2DAttrs const &);
OpTaskInvocation forward(Conv2DAttrs const &);
OpTaskInvocation backward(Conv2DAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  Conv2DAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* namespace Conv2DInput { */
/* static constexpr int INDEX = 0; */

/* enum { WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3, REPLICA = 4, NUMDIM };
 */
/* } */

/* namespace Conv2DOutput { */
/* enum { WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3, REPLICA = 4, NUMDIM };
 */
/* } */

/* namespace Conv2DKernel { */
/* static constexpr int INDEX = 0; */

/* enum { */
/*   WIDTH = 0, */
/*   HEIGHT = 1, */
/*   CHANNEL_IN = 2, */
/*   CHANNEL_OUT = 3, */
/*   REPLICA = 4, */
/*   NUMDIM */
/* }; */
/* } */

/* /1* namespace Conv2DBias { *1/ */
/* /1* static constexpr int INDEX = 1; *1/ */

/* class Conv2D : public Op { */
/* public: */
/*   Conv2D(FFModel &model, */
/*          LayerID const &layer_guid, */
/*          const ParallelTensor input, */
/*          int outChannels, */
/*          int kernelH, */
/*          int kernelW, */
/*          int strideH, */
/*          int strideW, */
/*          int paddingH, */
/*          int paddingW, */
/*          ActiMode activation, */
/*          int groups, */
/*          bool use_bias, */
/*          bool allocate_weights, */
/*          char const *name); */
/*   Conv2D(FFModel &model, */
/*          Conv2D const &other, */
/*          const ParallelTensor input, */
/*          bool allocate_weights); */
/*   Conv2D(FFModel &model, */
/*          Conv2DAttrs const &attrs, */
/*          std::vector<ParallelTensor> const &inputs, */
/*          char const *name = nullptr, */
/*          bool allocate_weights = false); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   // void update(const FFModel&); */
/*   // Parameter* get_parameter(int index); */
/*   // void create_weights(FFModel& model); */
/*   // void create_input_partition(FFModel& model); */
/*   static Op * */
/*       create_operator_from_layer(FFModel &model, */
/*                                  Layer const *layer, */
/*                                  std::vector<ParallelTensor> const &inputs);
 */

/*   static PerDeviceOpState *init_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
/*   static void forward_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
/*   static void backward_task(Legion::Task const *task, */
/*                             std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                             Legion::Context ctx, */
/*                             Legion::Runtime *runtime); */
/*   bool measure_operator_cost(Simulator *sim, */
/*                              MachineView const &pc, */
/*                              CostMetrics &cost_metrics) const override; */
/*   bool estimate_sync_cost(Simulator *sim, */
/*                           MachineView const &pc, */
/*                           CostMetrics &cost_metrics) const override; */

/*   /1* static void *1/ */
/*   /1*     construct_output_mappings(std::vector<ParallelDimMappingRecord> &);
 * *1/ */
/*   /1* static void construct_mappings(std::vector<ParallelDimMappingRecord> &,
 * *1/ */
/*   /1*                                bool use_bias); *1/ */
/*   /1* static void
 * construct_weight_mappings(std::vector<ParallelDimMappingRecord> &, *1/ */
/*   /1*                                       bool use_bias); *1/ */

/* public: */
/*   int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, */
/*       padding_h, padding_w; */
/*   ActiMode activation; */
/*   int groups; */
/*   bool use_bias; */
/* }; */

} // namespace FlexFlow

#endif
