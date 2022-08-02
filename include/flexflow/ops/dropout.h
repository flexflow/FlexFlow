#ifndef _FLEXFLOW_DROPOUT_H
#define _FLEXFLOW_DROPOUT_H

#include "flexflow/model.h"

namespace FlexFlow {

class DropoutMeta;

struct DropoutParams {
  float rate;
  unsigned long long seed;
  size_t get_hash(const ParallelTensor input) const;
};

class Dropout : public Op {
public:
  Dropout(FFModel &model,
          const ParallelTensor input,
          float rate,
          unsigned long long seed,
          char const *name);
  Dropout(FFModel &model, Dropout const &other, const ParallelTensor input);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override { assert(0); }
  static Op *
  create_operator_from_layer(FFModel &model,
                             Layer const *layer,
                             std::vector<ParallelTensor> const &inputs);

  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void forward_kernel(DropoutMeta *m,
                             float const *input_ptr,
                             float *output_ptr,
                             ffStream_t stream);
  static void forward_kernel_wrapper(DropoutMeta *m,
                                     float const *input_ptr,
                                     float *output_ptr);
  static void backward_kernel(DropoutMeta *m,
                              float const *output_grad_ptr,
                              float *input_grad_ptr,
                              ffStream_t stream);
  static void backward_kernel_wrapper(DropoutMeta *m,
                                      float const *output_grad_ptr,
                                      float *input_grad_ptr);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);

  size_t get_params_hash() const override;

  DropoutParams get_params() const;

public:
  float rate;
  unsigned long long seed;
};

class DropoutMeta : public OpMeta {
public:
  DropoutMeta(FFHandler handle,
              Dropout const *dropout,
              Legion::Memory gpu_mem,
              Legion::Domain const &output_domain);
  ~DropoutMeta(void);
  Realm::RegionInstance reserveInst;
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnDropoutDescriptor_t dropoutDesc;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor;
  miopenDropoutDescriptor_t dropoutDesc;
#endif
  void *reserveSpace, *dropoutStates;
  size_t reserveSpaceSize, dropoutStateSize;
};

}; // namespace FlexFlow

#endif
