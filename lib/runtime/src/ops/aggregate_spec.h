#ifndef _FLEXFLOW_AGGREGATE_SPEC_H_
#define _FLEXFLOW_AGGREGATE_SPEC_H_

#include "model.h"

namespace FlexFlow {

#define AGGREGATE_SPEC_MAX_K 4
#define AGGREGATE_SPEC_MAX_BATCH_SIZE 32
#define AGGREGATE_SPEC_MAX_N 12

class AggregateSpecMeta : public OpMeta {
public:
  AggregateSpecMeta(FFHandler handle, int n);
  ~AggregateSpecMeta(void);
  float **dev_region_ptrs;
};

class AggregateSpec : public Op {
public:
  using Params = AggregateSpecParams;
  using Input = ParallelTensor;
  AggregateSpec(FFModel &model,
                ParallelTensor const *inputs,
                int _n,
                float _lambda_bal,
                char const *name);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
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
  Params get_params() const;

public:
  int n;
  float lambda_bal;
};

}; // namespace FlexFlow
#endif
