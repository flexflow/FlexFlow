#ifndef _FLEXFLOW_ATTENTION_H
#define _FLEXFLOW_ATTENTION_H

#include "flexflow/model.h"

namespace FlexFlow {

class MultiHeadAttentionMeta;

class MultiHeadAttention : public Op {
public:
  MultiHeadAttention(FFModel& model,
                     const ParallelTensor _query,
                     const ParallelTensor _key,
                     const ParallelTensor _value,
                     int _embed_dim, int _num_heads,
                     int _kdim, int _vdim,
                     float _dropout, bool _bias,
                     bool _add_bias_kv, bool _add_zero_attn,
                     const char* name);
  MultiHeadAttention(FFModel& model,
                     const ParallelTensor _query,
                     const ParallelTensor _key,
                     const ParallelTensor _value,
                     const ParallelTensor _weight,
                     int _embed_dim, int _num_heads,
                     int _kdim, int _vdim,
                     float _dropout, bool _bias,
                     bool _add_bias_kv, bool _add_zero_attn,
                     const char* name);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void print_layer(const FFModel& model) override {assert(0);}
  bool get_int_parameter(PMParameter, int*) const override;

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const override;
  static void forward_kernel(const MultiHeadAttentionMeta* m,
                      const float* query_ptr,
                      const float* key_ptr,
                      const float* value_ptr,
                      const float* weight_ptr,
                      float* output_ptr,
                      ffStream_t stream);
  static void backward_kernel(const MultiHeadAttentionMeta* m,
                       const float* query_ptr,
                       float* query_grad_ptr,
                       const float* key_ptr,
                       float* key_grad_ptr,
                       const float* value_ptr,
                       float* value_grad_ptr,
                       const float* weight_ptr,
                       float* weight_grad_ptr,
                       const float* output_grad_ptr,
                       ffStream_t stream);            
  public:
  int num_heads;
  float dropout;
  bool bias;
  bool add_bias_kv, add_zero_attn;
  int qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize;
  int qoSeqLength, kvSeqLength;
};

class MultiHeadAttentionMeta : public OpMeta {
public:
  MultiHeadAttentionMeta(FFHandler handler,
                         const MultiHeadAttention* attn,
                         Legion::Memory gpu_mem,
                         int num_samples,
                         int num_heads);
  ~MultiHeadAttentionMeta(void);
public:
  Realm::RegionInstance reserveInst;
  size_t weightSize, reserveSpaceSize;
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  cudnnAttnDescriptor_t attnDesc;
  cudnnSeqDataDescriptor_t qDesc, kDesc, vDesc, oDesc;
#endif
  int *devQoSeqArray, *devKvSeqArray, *loWinIdx, *hiWinIdx;
  void *reserveSpace;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_ATTENTION_H
