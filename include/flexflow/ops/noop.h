#ifndef _FLEXFLOW_NOOP_H
#define _FLEXFLOW_NOOP_H

#include "flexflow/inference.h"
#include "flexflow/model.h"

namespace FlexFlow {

class NoOp : public Op {
public:
  NoOp(FFModel &model,
       OperatorType type,
       const ParallelTensor output,
       char const *name = NULL);
  NoOp(FFModel &model,
       OperatorType type,
       size_t input_tensor_guid,
       const ParallelTensor output,
       char const *name = NULL);
  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
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

  size_t get_params_hash() const override;
  tl::optional<RecordFormatter> as_dot() const override;

public:
  size_t input_tensor_guid;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_NOOP_H
