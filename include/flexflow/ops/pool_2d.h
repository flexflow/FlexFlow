#ifndef _FLEXFLOW_POOL_2D_H
#define _FLEXFLOW_POOL_2D_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/pool_2d_params.h"

namespace FlexFlow {

namespace Pool2DInput {
constexpr int NUMDIM = 5, WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3,
              REPLICA = 4;
};

namespace Pool2DOutput {
constexpr int NUMDIM = 5, WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3,
              REPLICA = 4;
};

class Pool2DMeta : public OpMeta {
public:
  Pool2DMeta(FFHandler handle);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor;
  miopenActivationDescriptor_t actiDesc;
  miopenPoolingDescriptor_t poolDesc;
#endif
  bool relu;
  char op_name[MAX_OPNAME];
};

class Pool2D : public Op {
public:
  using Params = Pool2DParams;
  using Input = ParallelTensor;

  Pool2D(FFModel &model,
         const ParallelTensor input,
         int kernelH,
         int kernelW,
         int strideH,
         int strideW,
         int paddingH,
         int paddingW,
         PoolType type,
         ActiMode activation,
         char const *name);
  Pool2D(FFModel &model, Pool2D const &other, ParallelTensor const input);
  Pool2D(FFModel &model,
         Params const &params,
         const Input input,
         char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void reset_idx(FFModel const &) override;
  void pipeinit(FFModel const &) override;
  void pipeforward(FFModel const &) override;
  void pipebackward(FFModel const &) override;
  void update(FFModel const &);
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);

  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void init_kernel(Pool2D const *pool,
                          Pool2DMeta *m,
                          int input_w,
                          int input_h,
                          int input_c,
                          int input_n,
                          int output_w,
                          int output_h,
                          int output_c,
                          int output_n,
                          int pad_h,
                          int pad_w);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void forward_kernel(Pool2DMeta const *m,
                             float const *input_ptr,
                             float *output_ptr,
                             ffStream_t stream);
  static void forward_kernel_wrapper(Pool2DMeta const *m,
                                     float const *input_ptr,
                                     float *output_ptr);
  static void backward_kernel(Pool2DMeta const *m,
                              float const *input_ptr,
                              float *input_grad_ptr,
                              float const *output_ptr,
                              float const *output_grad_ptr,
                              ffStream_t stream);
  static void backward_kernel_wrapper(Pool2DMeta const *m,
                                      float const *input_ptr,
                                      float *input_grad_ptr,
                                      float const *output_ptr,
                                      float const *output_grad_ptr);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);

  static void
      construct_output_mappings(std::vector<ParallelDimMappingRecord> &);

  Params get_params() const;

private:
  int output_size(ParallelDim output_dims[MAX_TENSOR_DIM]);

  void register_mappings();
  void register_output_mappings();

public:
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
  // int fwd_input_idx = 0;
  // int fwd_output_idx = 0;
  // int bwd_input_idx = 0;
  // int bwd_output_idx = 0;
  // int init_output_idx = 0;
};

}; // namespace FlexFlow

#endif //_FLEXFLOW_POOL_2D_H
