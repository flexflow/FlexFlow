#ifndef _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_H
#define _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/inference.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/inc_multihead_self_attention_params.h"
#include "flexflow/utils/memory_allocator.h"
#include "math.h"
#include <cfloat>
#include <complex>
#if defined(FF_USE_HIP_ROCM)
#include <hip/hip_complex.h>
#endif

namespace FlexFlow {

class IncMultiHeadSelfAttentionMeta;

class IncMultiHeadSelfAttention : public Op {
public:
  using Params = IncMultiHeadSelfAttentionParams;
  using Input = ParallelTensor;

  IncMultiHeadSelfAttention(FFModel &model,
                            LayerID const &layer_guid,
                            ParallelTensor const _input,
                            int _embed_dim,
                            int _num_q_heads,
                            int _num_kv_heads,
                            int _kdim,
                            int _vdim,
                            float _dropout,
                            bool _add_zero_attn,
                            RotaryEmbeddingMeta _rotary_embedding_meta,
                            bool _scaling_query,
                            float _scaling_factor,
                            bool _qk_prod_scaling,
                            bool _position_bias,
                            DataType _quantization_type,
                            bool _offload,
                            int _tensor_parallelism_degree,
                            char const *name);
  IncMultiHeadSelfAttention(FFModel &model,
                            ParallelTensor const _input,
                            int _embed_dim,
                            int _num_q_heads,
                            int _num_kv_heads,
                            int _kdim,
                            int _vdim,
                            float _dropout,
                            bool _add_zero_attn,
                            RotaryEmbeddingMeta _rotary_embedding_meta,
                            bool _scaling_query,
                            float _scaling_factor,
                            bool _qk_prod_scaling,
                            bool _position_bias,
                            DataType _quantization_type,
                            bool _offload,
                            int _tensor_parallelism_degree,
                            char const *name);
  IncMultiHeadSelfAttention(FFModel &model,
                            IncMultiHeadSelfAttention const &other,
                            ParallelTensor const input);
  IncMultiHeadSelfAttention(FFModel &model,
                            Params const &params,
                            Input const &inputs,
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
  Legion::FutureMap peft_bwd(FFModel const &,
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
  static void peft_bwd_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &mv,
                             CostMetrics &cost_metrics) const override;
  static void inference_kernel_wrapper(IncMultiHeadSelfAttentionMeta *m,
                                       BatchConfig const *bc,
                                       int shard_id,
                                       GenericTensorAccessorR const &input,
                                       GenericTensorAccessorW const &output);
  static void
      peft_bwd_kernel_wrapper(IncMultiHeadSelfAttentionMeta *m,
                              BatchConfig const *bc,
                              int shard_id,
                              GenericTensorAccessorW const &input_grad,
                              GenericTensorAccessorR const &output_grad);
  Params get_params() const;

public:
  int num_q_heads, num_kv_heads, tensor_parallelism_degree;
  float dropout, scaling_factor;
  bool add_zero_attn, scaling_query, qk_prod_scaling, position_bias;
  RotaryEmbeddingMeta rotary_embedding_meta;
  int qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize;
  int qoSeqLength, kvSeqLength;
  DataType quantization_type;
  bool offload;
};

class IncMultiHeadSelfAttentionMeta : public OpMeta {
public:
  IncMultiHeadSelfAttentionMeta(FFHandler handler,
                                IncMultiHeadSelfAttention const *attn,
                                MemoryAllocator &gpu_mem_allocator,
                                int num_samples,
                                int _num_q_heads,
                                int _num_kv_heads);
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
                                RotaryEmbeddingMeta _rotary_embedding_meta,
                                bool _scaling_query,
                                bool _qk_prod_scaling,
                                bool _position_bias,
                                float _scaling_factor,
                                MemoryAllocator &gpu_mem_allocator,
                                int num_samples,
                                int _global_num_q_heads,
                                int _global_num_kv_heads,
                                int _num_q_heads,
                                int _num_kv_heads,
                                DataType _quantization_type,
                                bool _offload);
  ~IncMultiHeadSelfAttentionMeta(void);

public:
  Realm::RegionInstance reserveInst;
  size_t reserveSpaceSize;
  int qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize;
  int global_num_q_heads, global_num_kv_heads, num_q_heads, num_kv_heads,
      hidden_size;
  RotaryEmbeddingMeta *rotary_embedding_meta;
  bool *scaling_query;
  bool *qk_prod_scaling;
  bool *position_bias;
  float scaling_factor;
  void *devQKVProjArray, *keyCache, *valueCache;
  void *qk_prods, *qk_prods_softmax;
  void *attn_heads;
  BatchConfig::PerTokenInfo *token_infos;
  BatchConfig::PerRequestInfo *request_infos;
  DataType quantization_type;
  bool offload;
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t qk_tensor;
  cuFloatComplex *complex_input;
#elif defined(FF_USE_HIP_ROCM)
  miopenTensorDescriptor_t qk_tensor;
  //  typedef hipFloatComplex attFloatComplex;
  hipFloatComplex *complex_input;
#endif
  // PEFT specific fields
  void *softmax_activation_buffer;
  void *query_activation_buffer;
  size_t allocated_peft_buffer_size1 = 0, allocated_peft_buffer_size2 = 0;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_ATTENTION_H
