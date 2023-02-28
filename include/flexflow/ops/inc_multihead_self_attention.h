#ifndef _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_H
#define _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_H


#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/inc_multihead_self_attention_params.h"
#include "flexflow/inference.h"

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
                 BatchConfig const &,
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
  static void inference_kernel(IncMultiHeadSelfAttentionMeta const *m,
                               float const *input_ptr,
                               float const *weight_ptr,
                               float *output_ptr,
                               ffStream_t stream);
  static void inference_kernel_wrapper(IncMultiHeadSelfAttentionMeta const *m,
                                     float const *input_ptr,
                                     float const *weight_ptr,
                                     float *output_ptr);
  Params get_params() const;
public:
  int num_heads;
  float dropout;
  bool bias;
  bool add_bias_kv, add_zero_attn;
  int qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize;
  int qoSeqLength, kvSeqLength;
};

class IncMultiHeadSelfAttentionMeta : public OpMeta {
public:
  IncMultiHeadSelfAttentionMeta(FFHandler handler,
                         IncMultiHeadSelfAttention const *attn,
                         Legion::Memory gpu_mem,
                         int num_samples,
                         int num_heads);
  ~IncMultiHeadSelfAttentionMeta(void);

public:
  Realm::RegionInstance reserveInst;
  size_t weightSize, reserveSpaceSize;
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnAttnDescriptor_t attnDesc;
  cudnnSeqDataDescriptor_t qDesc, kDesc, vDesc, oDesc;
#endif
  int *devQoSeqArray, *devKvSeqArray, *loWinIdx, *hiWinIdx;
  void *reserveSpace;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_ATTENTION_H
