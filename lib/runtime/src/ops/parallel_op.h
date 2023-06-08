#ifndef _FLEXFLOW_PARALLEL_OP_H
#define _FLEXFLOW_PARALLEL_OP_H

#include "parallel_op_info.h"
#include "utils/optional.h"

namespace FlexFlow {

struct ParallelOpJoinResult {
  optional<ParallelOpInfo> op = nullopt;
  bool join_did_succeed = false;
};

ParallelOpJoinResult try_join_parallel_ops(ParallelOpInfo const &,
                                           ParallelOpInfo const &);

/* class ParallelOp : public Op { */
/* public: */
/*   ParallelOp(FFModel &model, */
/*              OperatorType type, */
/*              char const *_name, */
/*              const ParallelTensor input); */
/*   virtual void init(FFModel const &) = 0; */
/*   virtual void forward(FFModel const &) = 0; */
/*   virtual void backward(FFModel const &) = 0; */
/*   virtual void create_input_partition(FFModel &model) = 0; */
/*   virtual bool measure_operator_cost(Simulator *sim, */
/*                                      MachineView const &pc, */
/*                                      CostMetrics &cost_metrics) const = 0; */
/*   virtual bool append_parallel_op_info( */
/*       std::vector<ParallelOpInfo> &parallel_ops) const = 0; */
/*   virtual bool is_parallel_op() const; */

/* public: */
/*   Legion::LogicalPartition input_lp, output_grad_lp; */
/* }; */

} // namespace FlexFlow

#endif
