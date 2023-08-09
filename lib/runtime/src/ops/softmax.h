#ifndef _FLEXFLOW_SOFTMAX_H
#define _FLEXFLOW_SOFTMAX_H

#include "op-attrs/ops/softmax.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<SOFTMAX_INIT_TASK_ID>();
template <>
void register_task<SOFTMAX_FWD_TASK_ID>();
template <>
void register_task<SOFTMAX_BWD_TASK_ID>();

OpTaskInvocation init(SoftmaxAttrs const &);
OpTaskInvocation forward(SoftmaxAttrs const &);
OpTaskInvocation backward(SoftmaxAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  SoftmaxAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Softmax : public Op { */
/* public: */
/*   Softmax(FFModel &model, */
/*           ParallelTensor const &logit, */
/*           int dim, */
/*           char const *name); */
/*   Softmax(FFModel &model, */
/*           SoftmaxAttrs const &attrs, */
/*           std::vector<ParallelTensor> const &input, */
/*           char const *name = nullptr); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
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

/* private: */
/*   template <int NDIM> */
/*   static void */
/*       forward_task_with_dim(Legion::Task const *task, */
/*                             std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                             Legion::Context ctx, */
/*                             Legion::Runtime *runtime); */
/*   template <int NDIM> */
/*   static void */
/*       backward_task_with_dim(Legion::Task const *task, */
/*                              std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                              Legion::Context ctx, */
/*                              Legion::Runtime *runtime); */

/* public: */
/*   int dim; */
/* }; */

} // namespace FlexFlow

#endif
