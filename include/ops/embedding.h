#ifndef _FLEXFLOW_EMBEDDING_H
#define _FLEXFLOW_EMBEDDING_H

#include "model.h"

class EmbeddingMeta : public OpMeta {
public:
  EmbeddingMeta(FFHandler handle): OpMeta(handle) {}
  AggrMode aggr;
};

class Embedding : public Op {
public:
  Embedding(FFModel& model,
            const Tensor _input,
            int _num_entries,
            int _out_channels,
            AggrMode _aggr,
            bool allocate_weights,
            const char* name);
  Embedding(FFModel& model,
            Embedding const &other,
            const Tensor input,
            bool allocate_weights);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
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
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
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

#endif // _FLEXFLOW_EMBEDDING_H
