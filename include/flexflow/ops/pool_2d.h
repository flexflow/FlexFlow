#ifndef _FLEXFLOW_POOL_2D_H
#define _FLEXFLOW_POOL_2D_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/pool_2d_params.h"

namespace FlexFlow {

namespace Pool2DInput {
constexpr int NUMDIM = 5, WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3,
              REPLICA = 4;
};

namespace Pool2DOutput {
constexpr int NUMDIM = 5, WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3,
              REPLICA = 4;
};

class Pool2D : public Op {
public:
  using Params = Pool2DParams;
  using Input = ParallelTensor;

  Pool2D(FFModel &model,
         const ParallelTensor input,
         int kernelH,
         int kernelW,
         int strideH,
         int strideW,
         int paddingH,
         int paddingW,
         PoolType type,
         ActiMode activation,
         char const *name);
  Pool2D(FFModel &model, Pool2D const &other, ParallelTensor const input);
  Pool2D(FFModel &model,
         Params const &params,
         const Input input,
         char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void update(FFModel const &);
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

  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);

  static void
      construct_output_mappings(std::vector<ParallelDimMappingRecord> &);

  Params get_params() const;

private:
  int output_size(ParallelDim output_dims[MAX_TENSOR_DIM]);

  void register_mappings();
  void register_output_mappings();

public:
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
};

}; // namespace FlexFlow

#endif //_FLEXFLOW_POOL_2D_H
