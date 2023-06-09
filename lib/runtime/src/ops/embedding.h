#ifndef _FLEXFLOW_EMBEDDING_H
#define _FLEXFLOW_EMBEDDING_H

#include "op-attrs/ops/embedding.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<EMBED_INIT_TASK_ID>();
template <>
void register_task<EMBED_FWD_TASK_ID>();
template <>
void register_task<EMBED_BWD_TASK_ID>();

OpTaskInvocation init(EmbeddingAttrs const &);
OpTaskInvocation forward(EmbeddingAttrs const &);
OpTaskInvocation backward(EmbeddingAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  EmbeddingAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* namespace Weight { */
/* enum { */
/*   OUT_CHANNELS = 0, */
/*   VOCAB_SIZE = 1, */
/* }; */
/* }; */

/* namespace Output { */
/* enum { OUT_CHANNELS = 0 }; */
/* }; */

/* class Embedding; */

/* class Embedding : public Op { */
/* public: */
/*   using Attrs = EmbeddingAttrs; */

/*   Embedding(FFModel &model, */
/*             LayerID const &_layer_guid, */
/*             const ParallelTensor _input, */
/*             int _num_entries, */
/*             int _out_channels, */
/*             AggrMode _aggr, */
/*             bool allocate_weights, */
/*             DataType _dtype, */
/*             char const *name); */
/*   Embedding(FFModel &model, */
/*             Embedding const &other, */
/*             const ParallelTensor input, */
/*             bool allocate_weights); */
/*   Embedding(FFModel &model, */
/*             Attrs const &params, */
/*             std::vector<ParallelTensor> const &input, */
/*             bool allocate_weights = false, */
/*             char const *name = nullptr); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   // void update(const FFModel&); */
/*   // Parameter* get_parameter(int index); */
/*   // void create_weights(FFModel& model); */
/*   // void create_input_partition(FFModel& model); */
/*   static Op * */
/*       create_operator_from_layer(FFModel &model, */
/*                                  Layer const *layer, */
/*                                  std::vector<ParallelTensor> const &inputs);
 */

/*   static PerDeviceOpState *init_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
/*   static void forward_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
/*   static void backward_task(Legion::Task const *task, */
/*                             std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                             Legion::Context ctx, */
/*                             Legion::Runtime *runtime); */
/*   static void */
/*       forward_task_cpu(Legion::Task const *task, */
/*                        std::vector<Legion::PhysicalRegion> const &regions, */
/*                        Legion::Context ctx, */
/*                        Legion::Runtime *runtime); */
/*   static void */
/*       backward_task_cpu(Legion::Task const *task, */
/*                         std::vector<Legion::PhysicalRegion> const &regions,
 */
/*                         Legion::Context ctx, */
/*                         Legion::Runtime *runtime); */

/*   bool measure_operator_cost(Simulator *sim, */
/*                              MachineView const &pc, */
/*                              CostMetrics &cost_metrics) const override; */

/* private: */
/*   int input_vocab_size_replica_dim() const; */
/*   int input_channel_out_replica_dim() const; */
/*   int output_vocab_size_replica_dim() const; */

/*   int output_size(ParallelDim output_dims[MAX_TENSOR_DIM]); */
/*   int weight_size(ParallelDim weights_dims[MAX_TENSOR_DIM]); */

/*   void register_mappings(); */
/*   void register_output_mappings(); */
/*   void register_weight_mappings(); */

/* public: */
/*   int num_entries, out_channels; */
/*   AggrMode aggr; */
/* }; */

} // namespace FlexFlow

#endif
