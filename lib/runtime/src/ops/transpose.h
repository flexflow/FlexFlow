#ifndef _FLEXFLOW_TRANSPOSE_H_
#define _FLEXFLOW_TRANSPOSE_H_

#include "layer.h"
#include "operator.h"
#include "kernels/transpose_kernels.h"

namespace FlexFlow {

class Transpose : public Op {
public:
  Transpose(FFModel &model,
            TransposeAttrs const &attrs,
            std::vector<ParallelTensor> const &inputs,
            char const *name = nullptr);
  Transpose(FFModel &model,
            ParallelTensor const &input,
            std::vector<int> const &perm,
            char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;

  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);

  static PerDeviceOpState *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  void init_meta(TransposePerDeviceState *m,
                 Legion::Domain const &in_domain,
                 Legion::Domain const &out_domain) const;
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
  void serialize(Legion::Serializer &s) const override;
  /* static PCG::Node deserialize(FFModel &ff, */
  /*                              Legion::Deserializer &d, */
  /*                              ParallelTensor inputs[], */
  /*                              int num_inputs); */
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;

public:
  stack_vector<int, MAX_TENSOR_DIM> perm;
};

}

#endif
