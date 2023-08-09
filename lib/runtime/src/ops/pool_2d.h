#ifndef _FLEXFLOW_POOL_2D_H
#define _FLEXFLOW_POOL_2D_H

#include "op-attrs/ops/pool_2d.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<POOL2D_INIT_TASK_ID>();
template <>
void register_task<POOL2D_FWD_TASK_ID>();
template <>
void register_task<POOL2D_BWD_TASK_ID>();

OpTaskInvocation init(Pool2DAttrs const &);
OpTaskInvocation forward(Pool2DAttrs const &);
OpTaskInvocation backward(Pool2DAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  Pool2DAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Pool2D : public Op { */
/* public: */
/*   Pool2D(FFModel &model, */
/*          const ParallelTensor input, */
/*          int kernelH, */
/*          int kernelW, */
/*          int strideH, */
/*          int strideW, */
/*          int paddingH, */
/*          int paddingW, */
/*          PoolType type, */
/*          ActiMode activation, */
/*          char const *name); */
/*   Pool2D(FFModel &model, Pool2D const &other, ParallelTensor const input); */
/*   Pool2D(FFModel &model, */
/*          Pool2DAttrs const &attrs, */
/*          std::vector<ParallelTensor> const &inputs, */
/*          char const *name = nullptr); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   void update(FFModel const &); */
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

/*   void serialize(Legion::Serializer &) const override; */
/*   /1* static PCG::Node deserialize(FFModel &ff, *1/ */
/*   /1*                              Legion::Deserializer &d, *1/ */
/*   /1*                              ParallelTensor inputs[], *1/ */
/*   /1*                              int num_inputs); *1/ */

/*   /1* static void *1/ */
/*   /1*     construct_output_mappings(std::vector<ParallelDimMappingRecord> &);
 * *1/ */

/* private: */
/*   int output_size(ParallelDim output_dims[MAX_TENSOR_DIM]); */

/*   void register_mappings(); */
/*   void register_output_mappings(); */

/* public: */
/*   int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w; */
/*   PoolType pool_type; */
/*   ActiMode activation; */
/* }; */

} // namespace FlexFlow

#endif
