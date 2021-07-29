#ifndef _FLEXFLOW_LINEAR_H
#define _FLEXFLOW_LINEAR_H

#include "model.h"
#include "node_cache.h"

class LinearMeta : public OpMeta {
public:
  LinearMeta(FFHandler handle, int batch_size);
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  const float *one_ptr;
  ActiMode activation;
  bool use_bias;
  char op_name[MAX_OPNAME];
};

class LinearParams {
public:
  int in_channels, out_channels;
  bool use_bias;
  ActiMode activation;

  bool is_valid(TensorShape const &input_shape) const;
  void solve_dims(const Tensor input,
                  ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,
                  ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const;
  void solve_dims(TensorShape const &input_shape, 
                  TensorShape& output_shape,
                  TensorShape& kernel_shape,
                  TensorShape& bias_shape) const;
  void solve_dims(TensorShape const &input_shape,
                  ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,
                  ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const;
  void construct_mappings(std::vector<ParallelDimMappingRecord>&, TensorShape const &) const;
  size_t get_hash(const Tensor input) const;

  enum NamedDimensions {
    INPUT_CHANNEL,
    INPUT_SAMPLE,
    INPUT_REPLICA,
    OUTPUT_CHANNEL,
    OUTPUT_SAMPLE,
    OUTPUT_REPLICA,
    KERNEL_CHANNEL_IN,
    KERNEL_CHANNEL_OUT,
    BIAS_CHANNEL_OUT 
  };

  std::unordered_map<NamedDimensions, int> get_dimension_names(TensorShape const &input_name) const;
private:
  void mark_replica_dims(TensorShape const &input_shape,
                         ParallelDim output_dims[MAX_TENSOR_DIM],
                         ParallelDim kernel_dims[MAX_TENSOR_DIM],
                         ParallelDim bias_dims[MAX_TENSOR_DIM]) const;
  void calculate_nonreplica_dim_sizes(TensorShape const &input_shape,
                                      ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                                      ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,
                                      ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const;
};

class Linear : public Op {
public:
  Linear(FFModel& model,
         const Tensor input,
         int out_dim,
         ActiMode activation,
         bool _use_bias,
         bool allocate_weights,
         const char* name);
  Linear(NodeCache& node_cache,
         const Tensor input,
         int out_dim,
         ActiMode activation,
         bool use_bias,
         bool allocate_weights,
         const char* name);
  Linear(FFModel& model, 
         Linear const &other, 
         Tensor const input, 
         bool allocate_weights);

  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model);
  bool get_int_parameter(PMParameter, int*) const;

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(const LinearMeta* m,
                      const float* input_ptr,
                      float* output_ptr,
                      const float* filter_ptr,
                      const float* bias_ptr,
                      int in_dim, int out_dim, int batch_size,
                      cudaStream_t stream);
  static void backward_kernel(const LinearMeta* m,
                       const float* input_ptr,
                       float* input_grad_ptr,
                       const float* output_ptr,
                       float* output_grad_ptr,
                       const float* kernel_ptr,
                       float* kernel_grad_ptr,
                       float* bias_ptr,
                       int in_dim, int out_dim, int batch_size,
                       cudaStream_t stream);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  bool estimate_sync_cost(Simulator* sim,
                          const MachineView& pc,
                          CostMetrics& cost_metrics) const override;
  ParallelConfig get_random_parallel_config(const FFModel& ff) const;
  bool is_valid_parallel_config(const FFModel& ff, const ParallelConfig& pc) const;

  void serialize(Legion::Serializer&) const override;
  static Node deserialize(FFModel &ff, Legion::Deserializer& d, Tensor inputs[], int num_inputs);

  size_t get_params_hash() const override;
private:
  Linear(int guid,
         bool profiling,
         const Tensor input,
         int out_dim,
         ActiMode activation,
         bool use_bias,
         bool allocate_weights,
         const char* name);

  template<int NDIM>
  static OpMeta* init_task_with_dim(const Legion::Task *task,
                                    const std::vector<Legion::PhysicalRegion> &regions,
                                    Legion::Context ctx, Legion::Runtime *runtime);
  template<int NDIM>
  static void forward_task_with_dim(const Legion::Task *task,
                                    const std::vector<Legion::PhysicalRegion> &regions,
                                    Legion::Context ctx, Legion::Runtime *runtime);
  template<int NDIM>
  static void backward_task_with_dim(const Legion::Task *task,
                                     const std::vector<Legion::PhysicalRegion> &regions,
                                     Legion::Context ctx, Legion::Runtime *runtime);
  static bool use_cudnn_activation(ActiMode mode);

  void register_mappings();
  void register_output_mappings();
  void register_weight_mappings();


  LinearParams get_params() const;
public:
  int in_channels, out_channels;
  Tensor replica;
  bool use_bias;
  ActiMode activation;
};

#endif // _FLEXLOW_LINEAR_H
