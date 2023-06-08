#ifndef _FLEXFLOW_ATTENTION_H
#define _FLEXFLOW_ATTENTION_H

#include "op-attrs/ops/attention.h"
#include "op_task_signature.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<ATTENTION_INIT_TASK_ID>();
template <>
void register_task<ATTENTION_FWD_TASK_ID>();
template <>
void register_task<ATTENTION_BWD_TASK_ID>();

OpTaskInvocation init(MultiHeadAttentionAttrs const &);
OpTaskInvocation forward(MultiHeadAttentionAttrs const &);
OpTaskInvocation backward(MultiHeadAttentionAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  MultiHeadAttentionAttrs const &attrs,
                                  ParallelTensorShape const &query_shape,
                                  ParallelTensorShape const &key_shape,
                                  ParallelTensorShape const &value_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv);

/* class MultiHeadAttention : public Op { */
/* public: */
/*   MultiHeadAttention(FFModel &model, */
/*                      LayerID const &layer_guid, */
/*                      const ParallelTensor _query, */
/*                      const ParallelTensor _key, */
/*                      const ParallelTensor _value, */
/*                      int _embed_dim, */
/*                      int _num_heads, */
/*                      int _kdim, */
/*                      int _vdim, */
/*                      float _dropout, */
/*                      bool _bias, */
/*                      bool _add_bias_kv, */
/*                      bool _add_zero_attn, */
/*                      bool allocate_weights, */
/*                      char const *name); */
/*   MultiHeadAttention(FFModel &model, */
/*                      const ParallelTensor _query, */
/*                      const ParallelTensor _key, */
/*                      const ParallelTensor _value, */
/*                      const ParallelTensor _weight, */
/*                      int _embed_dim, */
/*                      int _num_heads, */
/*                      int _kdim, */
/*                      int _vdim, */
/*                      float _dropout, */
/*                      bool _bias, */
/*                      bool _add_bias_kv, */
/*                      bool _add_zero_attn, */
/*                      bool allocate_weights, */
/*                      char const *name); */
/*   MultiHeadAttention(FFModel &model, */
/*                      MultiHeadAttention const &other, */
/*                      const ParallelTensor query, */
/*                      const ParallelTensor key, */
/*                      const ParallelTensor value, */
/*                      bool allocate_weights); */
/*   MultiHeadAttention(FFModel &model, */
/*                      ParallelTensor const &query, */
/*                      ParallelTensor const &key, */
/*                      ParallelTensor const &value, */
/*                      MultiHeadAttentionAttrs const &, */
/*                      bool allocate_weights = false, */
/*                      char const *name = nullptr); */
/*   static Op * */
/*       create_operator_from_layer(FFModel &model, */
/*                                  Layer const *layer, */
/*                                  std::vector<ParallelTensor> const &inputs);
 */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
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
/*   bool measure_operator_cost(Simulator *sim, */
/*                              MachineView const &mv, */
/*                              CostMetrics &cost_metrics) const override; */

/*   OpTaskBinding get_init_task_binding() const override; */
/*   TaskID get_init_task_id() const override; */
/*   OpTaskBinding get_fwd_task_binding() const override; */
/*   TaskID get_fwd_task_id() const override; */
/*   OpTaskBinding get_bwd_task_binding() const override; */
/*   TaskID get_bwd_task_id() const override; */
/* public: */
/*   MultiHeadAttentionAttrs attrs; */
/*   int qSize, kSize, vSize, qProjSize; */
/*   int qoSeqLength, kvSeqLength; */
/* }; */

/* template <> OpTaskSignature get_signature<ATTENTION_INIT_TASK_ID>(); */
/* template <> OpTaskSignature get_signature<ATTENTION_FWD_TASK_ID>(); */
/* template <> OpTaskSignature get_signature<ATTENTION_BWD_TASK_ID>(); */

} // namespace FlexFlow

#endif
