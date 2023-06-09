#ifndef _FLEXFLOW_FLAT_H
#define _FLEXFLOW_FLAT_H

#include "op-attrs/ops/flat.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<FLAT_INIT_TASK_ID>();
template <>
void register_task<FLAT_FWD_TASK_ID>();
template <>
void register_task<FLAT_BWD_TASK_ID>();

OpTaskInvocation init(FlatAttrs const &);
OpTaskInvocation forward(FlatAttrs const &);
OpTaskInvocation backward(FlatAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  FlatAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* namespace FlatInput { */
/* constexpr int NUMDIM = 5, WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3, */
/*               REPLICA = 4; */
/* } */

/* namespace FlatOutput { */
/* constexpr int NUMDIM = 3, CHANNEL = 0, SAMPLE = 1, REPLICA = 2; */
/* } */

/* class Flat : public Op { */
/* public: */
/*   Flat(FFModel &model, ParallelTensor const &input, char const *name); */
/*   Flat(FFModel &model, */
/*        FlatAttrs const &params, */
/*        std::vector<ParallelTensor> const &input, */
/*        char const *name = nullptr); */

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
/* }; */

} // namespace FlexFlow

#endif
