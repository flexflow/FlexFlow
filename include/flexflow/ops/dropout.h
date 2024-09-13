#ifndef _FLEXFLOW_DROPOUT_H
#define _FLEXFLOW_DROPOUT_H

#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/operator.h"
#include "flexflow/ops/dropout_params.h"
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include <curand.h>
#include <curand_kernel.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#endif

namespace FlexFlow {

class Dropout : public Op {
public:
  using Params = DropoutParams;
  using Input = ParallelTensor;
  Dropout(FFModel &model,
          const ParallelTensor input,
          float rate,
          unsigned long long seed,
          char const *name);
  Dropout(FFModel &model, Dropout const &other, const ParallelTensor input);
  Dropout(FFModel &model,
          Params const &params,
          Input const input,
          char const *name = nullptr);
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

  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);

  Params get_params() const;

public:
  float rate;
  unsigned long long seed;
};

}; // namespace FlexFlow

#endif
