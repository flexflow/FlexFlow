#ifndef _FLEXFLOW_CONCAT_H
#define _FLEXFLOW_CONCAT_H

#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/node.h"
#include "flexflow/device.h"
#include "flexflow/layer.h"

namespace FlexFlow {

struct ConcatParams {
  int axis;
  
  bool is_valid(const std::vector<ParallelTensorShape> &) const;
};

bool operator==(const ConcatParams&, const ConcatParams&);

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(FFHandler handle) : OpMeta(handle) {};
  int legion_axis;
  char op_name[MAX_OPNAME];
};

class Concat : public Op {
public:
  using Params = ConcatParams;
  using Input = std::vector<ParallelTensor>;

  Concat(FFModel& model,
         int n,
         const ParallelTensor* inputs,
         int axis,
         const char* name);
  Concat(FFModel& model,
         const ConcatParams& params,
         const std::vector<ParallelTensor>& inputs,
         const char* name = nullptr);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void reset_idx(const FFModel&) override;
  void pipeinit(const FFModel&)  override;
  void pipeforward(const FFModel&)  override;
  void pipebackward(const FFModel&)  override;
  bool get_int_parameter(PMParameter, int*) const override;
  void print_layer(const FFModel& model) override {assert(0);}
  static Op* create_operator_from_layer(
      FFModel& model,
      const Layer* layer,
      const std::vector<ParallelTensor>& inputs);
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  void init_meta(ConcatMeta *meta) const;
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(float* output,
                             float const * const *inputs,
                             int num_inputs,
                             int axis,
                             const Legion::Domain& out_domain,
                             const Legion::Domain* in_domain,
                             ffStream_t stream);
  static void forward_kernel_wrapper(const ConcatMeta *m,
                                     float* output,
                                     float const * const *inputs,
                                     int num_inputs,
                                     int axis,
                                     const Legion::Domain& out_domain,
                                     const Legion::Domain* in_domain);
  static void backward_kernel(const float* output_grad,
                              float** input_grads,
                              int num_inputs,
                              int axis,
                              const Legion::Domain& out_grad_domain,
                              const Legion::Domain* in_grad_domain,
                              ffStream_t stream);
  static void backward_kernel_wrapper(const ConcatMeta *m,
                                      const float* output_grad,
                                      float** input_grads,
                                      int num_inputs,
                                      int axis,
                                      const Legion::Domain& out_grad_domain,
                                      const Legion::Domain* in_grad_domain);
  bool measure_operator_cost(Simulator* sim,
                             const MachineView& pc,
                             CostMetrics& cost_metrics) const override;

  Params get_params() const;
public:
  int legion_axis;
  int fwd_output_idx = 0;
  int bwd_output_idx = 0;
  int fwd_input_idx[MAX_NUM_INPUTS];
  int bwd_input_idx[MAX_NUM_INPUTS];
};

}; // namespace FlexFlow

namespace std {
  template <>
  struct hash<FlexFlow::ConcatParams> {
    size_t operator()(const FlexFlow::ConcatParams&) const;
  };
}; // namespace std

#endif // _FLEXFLOW_CONCAT_H
