#ifndef _FLEXFLOW_BATCH_MATMUL_H
#define _FLEXFLOW_BATCH_MATMUL_H

#include "model.h"

class BatchMatmulMeta : public OpMeta {
public:
  BatchMatmulMeta(FFHandler handler);
  int a_seq_length_dim, b_seq_length_dim;
};

class BatchMatmul : public Op {
public:
  BatchMatmul(FFModel& model,
              const Tensor A,
              const Tensor B,
              int a_seq_length_dim,
              int b_seq_length_dim);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model);
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(const BatchMatmulMeta* meta,
                      float* o_ptr,
                      const float* a_ptr,
                      const float* b_ptr,
                      const float* c_ptr,
                      int m, int n, int k,
                      int batch,
                      cudaStream_t stream,
                      int a_seq_length_dim = -1,
                      int b_seq_length_dim = -1,
                      int seq_length = -1);
  static void backward_kernel(const BatchMatmulMeta* meta,
                       const float* o_ptr,
                       const float* o_grad_ptr,
                       const float* a_ptr,
                       float* a_grad_ptr,
                       const float* b_ptr,
                       float* b_grad_ptr,
                       float* c_grad_ptr,
                       int m, int n, int k, int batch,
                       cudaStream_t stream);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
private:
  template<int NDIM>
  void init_with_dim(const FFModel& ff);
  template<int NDIM>
  void forward_with_dim(const FFModel& ff);
  template<int NDIM>
  void backward_with_dim(const FFModel& ff);
public:
  int a_seq_length_dim, b_seq_length_dim;
};

#endif
