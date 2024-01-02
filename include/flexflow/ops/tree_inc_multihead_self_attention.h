#ifndef _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_H
#define _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/inference.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/tree_inc_multihead_self_attention_params.h"
#include "math.h"
#include <cfloat>
#include <complex>

namespace FlexFlow {

class TreeIncMultiHeadSelfAttentionMeta;

class TreeIncMultiHeadSelfAttention : public Op {
public:
  using Params = TreeIncMultiHeadSelfAttentionParams;
  using Input = ParallelTensor;

  TreeIncMultiHeadSelfAttention(FFModel &model,
                                LayerID const &layer_guid,
                                const ParallelTensor _input,
                                int _embed_dim,
                                int _num_q_heads,
                                int _num_kv_heads,
                                int _kdim,
                                int _vdim,
                                float _dropout,
                                bool _qkv_bias,
                                bool _final_bias,
                                bool _add_zero_attn,
                                bool _apply_rotary_embedding,
                                bool _scaling_query,
                                float _scaling_factor,
                                bool _qk_prod_scaling,
                                bool _position_bias,
                                bool allocate_weights,
                                DataType _quantization_type,
                                bool _offload,
                                int _tensor_parallelism_degree,
                                char const *name);
  TreeIncMultiHeadSelfAttention(FFModel &model,
                                const ParallelTensor _input,
                                const ParallelTensor _weight,
                                int _embed_dim,
                                int _num_q_heads,
                                int _num_kv_heads,
                                int _kdim,
                                int _vdim,
                                float _dropout,
                                bool _qkv_bias,
                                bool _final_bias,
                                bool _add_zero_attn,
                                bool _apply_rotary_embedding,
                                bool _scaling_query,
                                float _scaling_factor,
                                bool _qk_prod_scaling,
                                bool _position_bias,
                                bool allocate_weights,
                                DataType _quantization_type,
                                bool _offload,
                                int _tensor_parallelism_degree,
                                char const *name);
  TreeIncMultiHeadSelfAttention(FFModel &model,
                                TreeIncMultiHeadSelfAttention const &other,
                                const ParallelTensor input,
                                bool allocate_weights);
  TreeIncMultiHeadSelfAttention(FFModel &model,
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
                              BatchConfigFuture const &,
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

  static void inference_kernel_wrapper(TreeIncMultiHeadSelfAttentionMeta *m,
                                       TreeVerifyBatchConfig const *bc,
                                       int shard_id,
                                       GenericTensorAccessorR const &input,
                                       GenericTensorAccessorR const &weight,
                                       GenericTensorAccessorW const &output,
                                       GenericTensorAccessorR const &bias);

  Params get_params() const;

public:
  int num_q_heads, num_kv_heads, tensor_parallelism_degree;
  float dropout, scaling_factor;
  bool qkv_bias;
  bool final_bias, add_zero_attn, apply_rotary_embedding, scaling_query,
      qk_prod_scaling, position_bias;
  int qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize;
  int qoSeqLength, kvSeqLength;
  DataType quantization_type;
  bool offload;
};

class TreeIncMultiHeadSelfAttentionMeta : public IncMultiHeadSelfAttentionMeta {
public:
  TreeIncMultiHeadSelfAttentionMeta(FFHandler handler,
                                    TreeIncMultiHeadSelfAttention const *attn,
                                    GenericTensorAccessorR const &weight,
                                    MemoryAllocator &gpu_mem_allocator,
                                    int num_samples,
                                    int _num_q_heads,
                                    int _num_kv_heads);
  ~TreeIncMultiHeadSelfAttentionMeta(void);

public:
  int num_active_tokens;
  Realm::RegionInstance committed_token_reserve_inst;
  TreeVerifyBatchConfig::CommittedTokensInfo *committed_token_infos;
  bool *request_completed;
  BatchConfig::BitMask *causalMask;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_H
