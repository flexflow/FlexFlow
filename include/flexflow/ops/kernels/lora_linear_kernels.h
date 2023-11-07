#ifndef _FLEXFLOW_OPS_KERNELS_LORA_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LORA_LINEAR_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/lora_linear.h"

namespace FlexFlow {

struct LoraLinearWeight {
  void *w0_ptr, *w1_ptr, *w0_grad_ptr, *w1_grad_ptr;
  void *w0_state_ptr, *w1_state_ptr;
  int in_dim, out_dim, rank;
};

class LoraLinearMeta : public OpMeta {
public:
  LoraLinearMeta(FFHandler handle, LoraLinear const *li);
  ~LoraLinearMeta(void);
  char op_name[MAX_OPNAME];
  // PEFT related fields
  void *low_rank_activation;
  void *input_activation;
  std::unordered_map<PEFTModelID, LoraLinearWeight> model_weights;
};

namespace Kernels {
namespace LoraLinear {
void inference_kernel_wrapper(LoraLinearMeta *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output);
void peft_bwd_kernel_wrapper(LoraLinearMeta *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad);

namespace Internal {
template <typename DT>
void inference_kernel(LoraLinearMeta *m,
                      BatchConfig const *bc,
                      DT const *input_ptr,
                      DT *output_ptr,
                      int in_dim,
                      int out_dim,
                      ffStream_t stream);
template <typename DT>
void peft_bwd_kernel(LoraLinearMeta *m,
                     BatchConfig const *bc,
                     DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     int in_dim,
                     int out_dim,
                     ffStream_t stream);
} // namespace Internal
} // namespace LoraLinear
} // namespace Kernels
} // namespace FlexFlow
#endif // _FLEXFLOW_OPS_KERNELS_LORA_LINEAR_KERNELS_H
