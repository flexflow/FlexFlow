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
  DataType input_data_type;
  AggrMode aggr;
};

struct EmbeddingParams {
  int num_entries, out_channels;
  AggrMode aggr;

  size_t get_hash(const ParallelTensor input) const;
};

class Embedding : public Op {
public:
  Embedding(FFModel& model,
            const LayerID& _layer_guid,
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
  static Op* create_operator_from_layer(FFModel& model,
                                        const Layer* layer,
                                        const std::vector<ParallelTensor>& inputs);

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
  template<typename TI>
  static void forward_kernel(TI const *input_ptr,
                             float *output_ptr,
                             float const *weight_ptr,
                             int in_dim,
                             int out_dim,
                             int batch_size,
                             AggrMode aggr,
                             int outputSize,
                             ffStream_t stream);
  template<typename TI>
  static void forward_kernel_wrapper(const EmbeddingMeta *m,
                                     TI const *input_ptr,
                                     float *output_ptr,
                                     float const *weight_ptr,
                                     int in_dim,
                                     int out_dim,
                                     int batch_size,
                                     AggrMode aggr,
                                     int outputSize);
  template<typename TI>
  static void backward_kernel(TI const *input_ptr,
                              float const *output_ptr,
                              float *weight_grad_ptr,
                              int in_dim,
                              int out_dim,
                              int batch_size,
                              AggrMode aggr,
                              int outputSize,
                              ffStream_t stream);
  template<typename TI>
  static void backward_kernel_wrapper(const EmbeddingMeta *m,
                                      TI const *input_ptr,
                                      float const *output_ptr,
                                      float *weight_grad_ptr,
                                      int in_dim,
                                      int out_dim,
                                      int batch_size,
                                      AggrMode aggr,
                                      int outputSize);
  void rand_generate_int64_wrapper(int64_t* ptr, size_t size, int64_t p) const;
  bool measure_operator_cost(Simulator* sim,
                             const MachineView& pc,
                             CostMetrics& cost_metrics) const override;

  size_t get_params_hash() const override;  

  EmbeddingParams get_params() const;
private:
  template<typename TI>
  static void forward_task_with_type(
      const Legion::Task *task,
      const std::vector<Legion::PhysicalRegion> &regions,
      Legion::Context ctx, Legion::Runtime *runtime);
  template<typename TI>
  static void backward_task_with_type(
      const Legion::Task *task,
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
