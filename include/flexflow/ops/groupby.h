#ifndef _FLEXFLOW_GROUPBY_H_
#define _FLEXFLOW_GROUPBY_H_

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/node.h"
#include "flexflow/ops/groupby_params.h"

namespace FlexFlow {

class GroupByMeta : public OpMeta {
public:
  GroupByMeta(FFHandler handle, int n, float _alpha);
  ~GroupByMeta(void);
  float alpha;
  float **dev_region_ptrs;
};

class Group_by : public Op {
public:
  using Params = Group_byParams;
  using Input = std::pair<ParallelTensor, ParallelTensor>;
  Group_by(FFModel &model,
           const ParallelTensor _input,
           const ParallelTensor _assign,
           int _n,
           float _alpha,
           char const *name);
  Group_by(FFModel &model,
           Group_by const &other,
           const ParallelTensor input,
           const ParallelTensor assign);
  Group_by(FFModel &model,
           Params const &params,
           Input const &inputs,
           char const *name = nullptr);
  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
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
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  static void forward_kernel_wrapper(GroupByMeta const *m,
                                     float const *input,
                                     int const *exp_assign,
                                     float **outputs,
                                     int n, // num experts
                                     int k, // chosen experts
                                     int batch_size,
                                     int data_dim);
  static void backward_kernel_wrapper(GroupByMeta const *m,
                                      float *input_grad,
                                      int const *exp_assign,
                                      float **output_grads,
                                      int n, // num experts
                                      int k, // chosen experts
                                      int batch_size,
                                      int data_dim);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  Params get_params() const;

public:
  int n;
  float alpha;
};

}; // namespace FlexFlow

#endif
