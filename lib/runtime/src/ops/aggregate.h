#ifndef _FLEXFLOW_AGGREGATE_H_
#define _FLEXFLOW_AGGREGATE_H_

#include "op_meta.h"
#include "operator.h"
#include "layer.h"

namespace FlexFlow {

#define AGGREGATE_MAX_K 4
#define AGGREGATE_MAX_BATCH_SIZE 64
#define AGGREGATE_MAX_N 12

class AggregateMeta : public OpMeta {
public:
  AggregateMeta(FFHandler handle, int n);
  ~AggregateMeta(void);
  float **dev_exp_preds;
  float **dev_exp_grads;
};

class Aggregate : public Op {
public:
  Aggregate(FFModel &model,
            ParallelTensor const *inputs,
            int _n,
            float _lambda_bal,
            char const *name);
  Aggregate(FFModel &model,
            Aggregate const &other,
            std::vector<ParallelTensor> const &inputs);
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
  void serialize(Legion::Serializer &s) const override;
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &mv,
                             CostMetrics &cost_metrics) const override;
  /* Params get_params() const; */

public:
  int n;
  float lambda_bal;
};

}; // namespace FlexFlow

#endif
