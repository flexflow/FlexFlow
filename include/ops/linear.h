#ifndef _FLEXFLOW_LINEAR_H
#define _FLEXFLOW_LINEAR_H

#include "model.h"

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

class Linear : public Op {
public:
  Linear(FFModel& model,
         const Tensor input,
         int out_dim,
         ActiMode activation,
         bool _use_bias,
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
                      int in_dim, int out_dim, int batch_size);
  static void backward_kernel(const LinearMeta* m,
                       const float* input_ptr,
                       float* input_grad_ptr,
                       const float* output_ptr,
                       float* output_grad_ptr,
                       const float* kernel_ptr,
                       float* kernel_grad_ptr,
                       float* bias_ptr,
                       int in_dim, int out_dim, int batch_size);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  bool estimate_sync_cost(Simulator* sim,
                          const MachineView& pc,
                          CostMetrics& cost_metrics) const;
  ParallelConfig get_random_parallel_config(const FFModel& ff) const;
  bool is_valid_parallel_config(const FFModel& ff, const ParallelConfig& pc) const;

/* #ifndef __CUDACC__ */
  void serialize(Legion::Serializer&) const override;
  static Node deserialize(FFModel &ff, Legion::Deserializer& d, Tensor inputs[], int num_inputs);
/* #endif */
private:
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

  int output_replica_dim() const;
  int output_channel_dim() const;
  int input_replica_dim() const;
  int input_channel_dim() const;

  int output_size(ParallelDim output_dims[MAX_TENSOR_DIM]);
  int kernel_size(ParallelDim kernel_dims[MAX_TENSOR_DIM]);
  int bias_size(ParallelDim bias_dims[MAX_TENSOR_DIM]);

  void register_mappings();
  void register_output_mappings();
  void register_weight_mappings();
public:
  int in_channels, out_channels;
  Tensor replica;
  bool use_bias;
  ActiMode activation;
};

#endif // _FLEXLOW_LINEAR_H
