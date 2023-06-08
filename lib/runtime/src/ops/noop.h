#ifndef _FLEXFLOW_NOOP_H
#define _FLEXFLOW_NOOP_H

#include "op-attrs/ops/input.h"
#include "op-attrs/ops/noop.h"
#include "op_task_invocation.h"

namespace FlexFlow {

template <>
void register_task<NOOP_INIT_TASK_ID>();

OpTaskInvocation init(NoopAttrs const &);
OpTaskInvocation forward(NoopAttrs const &);
OpTaskInvocation backward(NoopAttrs const &);

OpTaskInvocation init(InputAttrs const &);
OpTaskInvocation forward(InputAttrs const &);
OpTaskInvocation backward(InputAttrs const &);

/* class NoOp : public Op { */
/* public: */
/*   NoOp(FFModel &model, */
/*        OperatorType type, */
/*        ParallelTensor const &output, */
/*        char const *name = NULL); */
/*   NoOp(FFModel &model, */
/*        OperatorType type, */
/*        size_t input_tensor_guid, */
/*        ParallelTensor const &output, */
/*        char const *name = NULL); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   bool measure_operator_cost(Simulator *sim, */
/*                              MachineView const &pc, */
/*                              CostMetrics &cost_metrics) const override; */
/*   static PerDeviceOpState *init_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */

/*   tl::optional<RecordFormatter> as_dot() const override; */

/*   OpTaskBinding get_init_task_binding() const override; */
/*   TaskID get_init_task_id() const override; */
/*   OpTaskBinding get_fwd_task_binding() const override { assert(false); } */
/*   TaskID get_fwd_task_id() const override { assert(false); } */
/*   OpTaskBinding get_bwd_task_binding() const override { assert(false); } */
/*   TaskID get_bwd_task_id() const override { assert(false); } */
/* public: */
/*   size_t input_tensor_guid; */
/* }; */

} // namespace FlexFlow

#endif
