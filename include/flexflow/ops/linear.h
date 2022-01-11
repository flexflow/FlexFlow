#ifndef _FLEXFLOW_LINEAR_H
#define _FLEXFLOW_LINEAR_H

#include "flexflow/model.h"
#include "flexflow/node_cache.h"

namespace FlexFlow {

class LinearMeta : public OpMeta {
public:
  LinearMeta(FFHandler handle, int batch_size);
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#else
  miopenTensorDescriptor_t outputTensor;
  miopenActivationDescriptor_t actiDesc;
#endif
  const float *one_ptr;
  ActiMode activation;
  bool use_bias;
  DataType input_type, weight_type, output_type;
  char op_name[MAX_OPNAME];
};

class LinearParams {
public:
  LayerID layer_guid;
  int in_channels, out_channels;
  bool use_bias;
  DataType data_type;
  ActiMode activation;

  bool is_valid(ParallelTensorShape const &input_shape) const;
  void solve_dims(const ParallelTensor input,
                  ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,
                  ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const;
  void solve_dims(ParallelTensorShape const &input_shape, 
                  ParallelTensorShape& output_shape,
                  ParallelTensorShape& kernel_shape,
                  ParallelTensorShape& bias_shape) const;
  void solve_dims(ParallelTensorShape const &input_shape,
                  ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,
                  ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const;
  void construct_mappings(std::vector<ParallelDimMappingRecord>&, ParallelTensorShape const &) const;
  size_t get_hash(const ParallelTensor input) const;

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

  std::unordered_map<NamedDimensions, int> get_dimension_names(ParallelTensorShape const &input_name) const;
private:
  void mark_replica_dims(ParallelTensorShape const &input_shape,
                         ParallelDim output_dims[MAX_TENSOR_DIM],
                         ParallelDim kernel_dims[MAX_TENSOR_DIM],
                         ParallelDim bias_dims[MAX_TENSOR_DIM]) const;
  void calculate_nonreplica_dim_sizes(ParallelTensorShape const &input_shape,
                                      ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                                      ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,
                                      ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const;
};

class Linear : public Op {
public:
  Linear(FFModel& model,
         const LayerID& layer_guid,
         const ParallelTensor input,
         int out_dim,
         ActiMode activation,
         bool _use_bias,
         DataType _data_type,
         bool allocate_weights,
         const char* name);
  Linear(NodeCache& node_cache,
         const ParallelTensor input,
         int out_dim,
         ActiMode activation,
         bool use_bias,
	 DataType _data_type,
         bool allocate_weights,
         const char* name);
  Linear(FFModel& model, 
         Linear const &other, 
         ParallelTensor const input, 
         bool allocate_weights);

  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void print_layer(const FFModel& model) override;
  bool get_int_parameter(PMParameter, int*) const override;
  static Op* create_operator_from_layer(FFModel& model,
                                        const Layer* layer,
                                        const std::vector<ParallelTensor>& inputs);
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void init_kernel(LinearMeta *m,
                          int batch_size,
                          int channel);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(const LinearMeta* m,
                             const void* input_ptr,
                             void* output_ptr,
                             const void* filter_ptr,
                             const void* bias_ptr,
                             int in_dim, int out_dim, int batch_size,
                             ffStream_t stream);
  static void forward_kernel_wrapper(const LinearMeta* m,
                                     const void* input_ptr,
                                     void* output_ptr,
                                     const void* filter_ptr,
                                     const void* bias_ptr,
                                     int in_dim, int out_dim, int batch_size);
  static void backward_kernel(const LinearMeta* m,
                              const void* input_ptr,
                              void* input_grad_ptr,
                              const void* output_ptr,
                              void* output_grad_ptr,
                              const void* kernel_ptr,
                              void* kernel_grad_ptr,
                              void* bias_ptr,
                              int in_dim, int out_dim, int batch_size,
                              ffStream_t stream);
  static void backward_kernel_wrapper(const LinearMeta* m,
                                      const void* input_ptr,
                                      void* input_grad_ptr,
                                      const void* output_ptr,
                                      void* output_grad_ptr,
                                      const void* kernel_ptr,
                                      void* kernel_grad_ptr,
                                      void* bias_ptr,
                                      int in_dim, int out_dim, int batch_size);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const override;
  bool estimate_sync_cost(Simulator* sim,
                          const MachineView& pc,
                          CostMetrics& cost_metrics) const override;
  ParallelConfig get_random_parallel_config(const FFModel& ff) const override;
  bool is_valid_parallel_config(const FFModel& ff, const ParallelConfig& pc) const override;

  void serialize(Legion::Serializer&) const override;
  static PCG::Node deserialize(FFModel &ff, Legion::Deserializer& d, ParallelTensor inputs[], int num_inputs);

  size_t get_params_hash() const override;
private:
  Linear(int guid,
         bool profiling,
         const ParallelTensor input,
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
  static bool use_activation(ActiMode mode);

  void register_mappings();
  void register_output_mappings();
  void register_weight_mappings();


  LinearParams get_params() const;
public:
  int in_channels, out_channels;
  ActiMode activation;
  bool use_bias;
  ParallelTensor replica;
};

}; // namespace FlexFlow

#endif // _FLEXLOW_LINEAR_H
