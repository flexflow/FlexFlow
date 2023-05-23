#ifndef _FLEXFLOW_SPEC_INC_MULTIHEAD_SELF_ATTENTION_H
#define _FLEXFLOW_SPEC_INC_MULTIHEAD_SELF_ATTENTION_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/inference.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/spec_inc_multihead_self_attention_params.h"
#include "math.h"
#include <cfloat>
#include <complex>

namespace FlexFlow {

class SpecIncMultiHeadSelfAttentionMeta;

class SpecIncMultiHeadSelfAttention : public Op {
public:
  using Params = SpecIncMultiHeadSelfAttentionParams;
  using Input = ParallelTensor;

  SpecIncMultiHeadSelfAttention(FFModel &model,
                                LayerID const &layer_guid,
                                const ParallelTensor _input,
                                int _embed_dim,
                                int _num_heads,
                                int _kdim,
                                int _vdim,
                                float _dropout,
                                bool _bias,
                                bool _add_bias_kv,
                                bool _add_zero_attn,
                                bool _apply_rotary_embedding,
                                bool _scaling_query,
                                float _scaling_factor,
                                bool _qk_prod_scaling,
                                bool allocate_weights,
                                char const *name);
  SpecIncMultiHeadSelfAttention(FFModel &model,
                                const ParallelTensor _input,
                                const ParallelTensor _weight,
                                int _embed_dim,
                                int _num_heads,
                                int _kdim,
                                int _vdim,
                                float _dropout,
                                bool _bias,
                                bool _add_bias_kv,
                                bool _add_zero_attn,
                                bool _apply_rotary_embedding,
                                bool _scaling_query,
                                float _scaling_factor,
                                bool _qk_prod_scaling,
                                bool allocate_weights,
                                char const *name);
  SpecIncMultiHeadSelfAttention(FFModel &model,
                                SpecIncMultiHeadSelfAttention const &other,
                                const ParallelTensor input,
                                bool allocate_weights);
  SpecIncMultiHeadSelfAttention(FFModel &model,
                                Params const &params,
                                Input const &inputs,
                                bool allocate_weights = false,
                                char const *name = nullptr);
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfig const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  bool get_int_parameter(PMParameter, int *) const override;

  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &mv,
                             CostMetrics &cost_metrics) const override;

  static void
      inference_kernel_wrapper(SpecIncMultiHeadSelfAttentionMeta const *m,
                               BeamSearchBatchConfig const *bc,
                               float const *input_ptr,
                               float const *weight_ptr,
                               float *output_ptr,
                               float const *bias_ptr);
  Params get_params() const;

public:
  int num_heads;
  float dropout, scaling_factor;
  bool bias;
  bool add_bias_kv, add_zero_attn, apply_rotary_embedding, scaling_query,
      qk_prod_scaling;
  int qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize;
  int qoSeqLength, kvSeqLength;
};

class SpecIncMultiHeadSelfAttentionMeta : public IncMultiHeadSelfAttentionMeta {
public:
  SpecIncMultiHeadSelfAttentionMeta(FFHandler handler,
                                    SpecIncMultiHeadSelfAttention const *attn,
                                    float const *weight_ptr,
                                    Legion::Memory gpu_mem,
                                    int num_samples,
                                    int _num_heads);
  ~SpecIncMultiHeadSelfAttentionMeta(void);

public:
  Realm::RegionInstance beam_search_reserve_inst;
  BatchConfig::PerRequestInfo *request_infos;
  BeamSearchBatchConfig::BeamSearchPerTokenInfo *beam_token_infos;
  BeamSearchBatchConfig::BeamSearchPerRequestInfo *beam_request_infos;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_SPEC_INC_MULTIHEAD_SELF_ATTENTION_H
