#ifndef _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/lora_linear.h"

namespace FlexFlow {

class LoraLinearMeta : public OpMeta {
public:
  LoraLinearMeta(FFHandler handle,
                 LoraLinear const *li);
  ~LoraLinearMeta(void);
  char op_name[MAX_OPNAME];
  // PEFT related fields
  void *low_rank_activation;
  void *input_activation;
};

namespace Kernels {
namespace LoraLinear {
void inference_kernel_wrapper(LoraLinearMeta *m,
                              void const *input_ptr,
                              void *output_ptr,
                              void const *weight_first_ptr,
                              void const *weight_second_ptr,
                              int in_dim,
                              int out_dim,
                              int rank,
                              int num_infr_tokens,
                              int num_peft_tokens);
void peft_bwd_kernel_wrapper(LoraLinearMeta *m,
                             void *input_grad_ptr,
                             void const *output_grad_ptr,
                             void const *weight_first_ptr,
                             void const *weight_second_ptr,
                             void *weight_first_grad_ptr,
                             void *weight_second_grad_ptr,
                             int in_dim,
                             int out_dim,
                             int rank,
                             int num_infr_tokens,
                             int num_peft_tokens);
bool use_activation(ActiMode mode);

namespace Internal {
template <typename DT>
void inference_kernel(LoraLinearMeta *m,
                      void const *input_ptr,
                      void *output_ptr,
                      void const *weight_first_ptr,
                      void const *weight_second_ptr,
                      int in_dim,
                      int out_dim,
                      int rank,
                      int num_infr_tokens,
                      int num_peft_tokens,
                      ffStream_t stream);
template <typename DT>
void peft_bwd_kernel(LoraLinearMeta *m,
                     void *input_grad_ptr,
                     void const *output_grad_ptr,
                     void const *weight_first_ptr,
                     void const *weight_second_ptr,
                     void *weight_first_grad_ptr,
                     void *weight_second_grad_ptr,
                     int in_dim,
                     int out_dim,
                     int rank,
                     int num_infr_tokens,
                     int num_peft_tokens,
                     ffStream_t stream);
} // namespace Internal
} // namespace LoraLinear
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
