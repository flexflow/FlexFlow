#ifndef _FLEXFLOW_EMBEDDING_H
#define _FLEXFLOW_EMBEDDING_H

#include "flexflow/model.h"

namespace FlexFlow {
  
namespace Weight {
  enum {
    OUT_CHANNELS = 0,
    VOCAB_SIZE = 1,
  };
};

namespace Output {
  enum {
    OUT_CHANNELS = 0
  };
};

class EmbeddingMeta : public OpMeta {
public:
  EmbeddingMeta(FFHandler handle): OpMeta(handle) {}
  AggrMode aggr;
};

class Embedding : public Op {
public:
  Embedding(FFModel& model,
            const ParallelTensor _input,
            int _num_entries,
            int _out_channels,
            AggrMode _aggr,
            bool allocate_weights,
            const char* name);
  Embedding(FFModel& model,
            Embedding const &other,
            const ParallelTensor input,
            bool allocate_weights);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  //void update(const FFModel&);
  void print_layer(const FFModel& model) override {assert(0);}
  //Parameter* get_parameter(int index);
  //void create_weights(FFModel& model);
  //void create_input_partition(FFModel& model);

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task_cpu(const Legion::Task *task,
                               const std::vector<Legion::PhysicalRegion> &regions,
                               Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task_cpu(const Legion::Task *task,
                                const std::vector<Legion::PhysicalRegion> &regions,
                                Legion::Context ctx, Legion::Runtime *runtime);
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  static void forward_kernel(int64_t const *input_ptr,
                             float *output_ptr,
                             float const *weight_ptr,
                             int in_dim,
                             int out_dim,
                             int batch_size,
                             AggrMode aggr,
                             int outputSize,
                             cudaStream_t stream);
  static void backward_kernel(int64_t const *input_ptr,
                              float const *output_ptr,
                              float *weight_grad_ptr,
                              int in_dim,
                              int out_dim,
                              int batch_size,
                              AggrMode aggr,
                              int outputSize,
                              cudaStream_t stream);
#else
  static void forward_kernel(int64_t const *input_ptr,
                             float *output_ptr,
                             float const *weight_ptr,
                             int in_dim,
                             int out_dim,
                             int batch_size,
                             AggrMode aggr,
                             int outputSize,
                             hipStream_t stream);
  static void backward_kernel(int64_t const *input_ptr,
                              float const *output_ptr,
                              float *weight_grad_ptr,
                              int in_dim,
                              int out_dim,
                              int batch_size,
                              AggrMode aggr,
                              int outputSize,
                              hipStream_t stream);
#endif
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const override;
private:
  template<int NDIM>
  static void forward_task_with_dim(const Legion::Task *task,
                                    const std::vector<Legion::PhysicalRegion> &regions,
                                    Legion::Context ctx, Legion::Runtime *runtime);
  template<int NDIM>
  static void backward_task_with_dim(const Legion::Task *task,
                                     const std::vector<Legion::PhysicalRegion> &regions,
                                     Legion::Context ctx, Legion::Runtime *runtime);

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
