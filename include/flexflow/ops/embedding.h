#ifndef _FLEXFLOW_EMBEDDING_H
#define _FLEXFLOW_EMBEDDING_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/embedding_params.h"

namespace FlexFlow {

namespace Weight {
enum {
  OUT_CHANNELS = 0,
  VOCAB_SIZE = 1,
};
};

namespace Output {
enum { OUT_CHANNELS = 0 };
};

class Embedding;

class Embedding : public Op {
public:
  using Params = EmbeddingParams;
  using Input = ParallelTensor;

  Embedding(FFModel &model,
            LayerID const &_layer_guid,
            const ParallelTensor _input,
            int _num_entries,
            int _out_channels,
            AggrMode _aggr,
            bool allocate_weights,
            DataType _dtype,
            char const *name);
  Embedding(FFModel &model,
            Embedding const &other,
            const ParallelTensor input,
            bool allocate_weights);
  Embedding(FFModel &model,
            Params const &params,
            Input const input,
            bool allocate_weights = false,
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
  // void update(const FFModel&);
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  // Parameter* get_parameter(int index);
  // void create_weights(FFModel& model);
  // void create_input_partition(FFModel& model);
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
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void
      forward_task_cpu(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static void
      backward_task_cpu(Legion::Task const *task,
                        std::vector<Legion::PhysicalRegion> const &regions,
                        Legion::Context ctx,
                        Legion::Runtime *runtime);

  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  Params get_params() const;

private:
#ifdef DEADCODE
  template <typename TI>
  static void
      forward_task_with_type(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  template <typename TI>
  static void backward_task_with_type(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
#endif
  int input_vocab_size_replica_dim() const;
  int input_channel_out_replica_dim() const;
  int output_vocab_size_replica_dim() const;

  int output_size(ParallelDim output_dims[MAX_TENSOR_DIM]);
  int weight_size(ParallelDim weights_dims[MAX_TENSOR_DIM]);

  void register_mappings();
  void register_output_mappings();
  void register_weight_mappings();

public:
  int num_entries, out_channels;
  AggrMode aggr;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_EMBEDDING_H
