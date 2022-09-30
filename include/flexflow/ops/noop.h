#ifndef _FLEXFLOW_NOOP_H
#define _FLEXFLOW_NOOP_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/params/noop_params.h"

namespace FlexFlow {

class NoOp : public Op {
public:
  using Params = NoOpParams;

  NoOp(FFModel &model,
       OperatorType type,
       const ParallelTensor output,
       char const *name = NULL);
  NoOp(FFModel &model,
       OperatorType type,
       size_t input_tensor_guid,
       const ParallelTensor output,
       char const *name = NULL);
  NoOp(FFModel &model,
       Params const &params,
       std::vector<ParallelTensor> const &inputs,
       char const *name = nullptr);
      
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);

  tl::optional<RecordFormatter> as_dot() const override;

  Params get_params() const;
public:
  size_t input_tensor_guid;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_NOOP_H
