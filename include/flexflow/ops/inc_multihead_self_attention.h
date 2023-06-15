#ifndef _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_H
#define _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/inference.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/inc_multihead_self_attention_params.h"
#include "math.h"
#include <cfloat>
#include <complex>

namespace FlexFlow {

class IncMultiHeadSelfAttentionMeta;

class IncMultiHeadSelfAttention : public Op {
public:
  using Params = IncMultiHeadSelfAttentionParams;
  using Input = ParallelTensor;

  IncMultiHeadSelfAttention(FFModel &model,
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
                            bool _output_bias,
                            bool allocate_weights,
                            char const *name);
  IncMultiHeadSelfAttention(FFModel &model,
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
                            bool _output_bias,
                            bool allocate_weights,
                            char const *name);
  IncMultiHeadSelfAttention(FFModel &model,
                            IncMultiHeadSelfAttention const &other,
                            const ParallelTensor input,
                            bool allocate_weights);
  IncMultiHeadSelfAttention(FFModel &model,
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
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &mv,
                             CostMetrics &cost_metrics) const override;

  static void inference_kernel_wrapper(IncMultiHeadSelfAttentionMeta const *m,
                                       BatchConfig const *bc,
                                       GenericTensorAccessorR const &input,
                                       GenericTensorAccessorR const &weight,
                                       GenericTensorAccessorW const &output,
                                       GenericTensorAccessorR const &bias);
  Params get_params() const;

public:
  int num_heads;
  float dropout, scaling_factor;
  bool bias;
  bool add_bias_kv, add_zero_attn, apply_rotary_embedding, scaling_query,
      qk_prod_scaling;
  int qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize;
  int qoSeqLength, kvSeqLength;
  bool output_bias;
};

class IncMultiHeadSelfAttentionMeta : public OpMeta {
public:
  IncMultiHeadSelfAttentionMeta(FFHandler handler,
                                IncMultiHeadSelfAttention const *attn,
                                GenericTensorAccessorR const &weight,
                                Legion::Memory gpu_mem,
                                int num_samples,
                                int _num_heads);
  IncMultiHeadSelfAttentionMeta(FFHandler handler,
                                InferenceMode infer_mode,
                                Op const *attn,
                                int _qSize,
                                int _kSize,
                                int _vSize,
                                int _qProjSize,
                                int _kProjSize,
                                int _vProjSize,
                                int _oProjSize,
                                bool _apply_rotary_embedding,
                                bool _bias,
                                bool _scaling_query,
                                bool _qk_prod_scaling,
                                bool _add_bias_kv,
                                float _scaling_factor,
                                GenericTensorAccessorR const &weight,
                                Legion::Memory gpu_mem,
                                int num_samples,
                                int _num_heads,
                                bool _output_bias);
  ~IncMultiHeadSelfAttentionMeta(void);

public:
  Realm::RegionInstance reserveInst;
  size_t weights_params, weightSize, reserveSpaceSize;
  int qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize;
  int num_heads;
  bool *has_load_weights;
  bool *apply_rotary_embedding;
  bool *bias;
  bool *scaling_query;
  bool *qk_prod_scaling;
  float scaling_factor;
  bool *output_bias;
#ifdef INFERENCE_TESTS
  float *kcache, *vcache;
#endif
  void *devQKVProjArray, *keyCache, *valueCache;
  void *qk_prods, *qk_prods_softmax;
  void *attn_heads, *W_out_contiguous;
  BatchConfig::PerTokenInfo *token_infos;
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cuFloatComplex *complex_input;
#endif
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_ATTENTION_H
