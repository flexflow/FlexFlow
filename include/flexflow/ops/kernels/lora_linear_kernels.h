#ifndef _FLEXFLOW_OPS_KERNELS_LORA_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LORA_LINEAR_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/lora_linear.h"

namespace FlexFlow {
using Legion::Context;
using Legion::Runtime;
struct LoraLinearWeight {
  // weights
  void *w0_ptr, *w1_ptr;
  // gradients
  void *w0_grad_ptr, *w1_grad_ptr;
  // v values for SGD optimizer (when using momentum)
  void *w0_v_values_ptr, *w1_v_values_ptr;
  int in_dim, out_dim, rank, num_shards;
};

struct LoraLinearModelState {
  LoraLinearWeight weights;
  LoraOptimizerConfig const *optimizer_config;
  float lora_alpha;
  std::string cache_folder;
  // Huggingface model ID (for download and/or upload)
  std::string peft_model_id;
};

class LoraLinearMeta : public OpMeta {
public:
  LoraLinearMeta(FFHandler handle, LoraLinear const *li);
  ~LoraLinearMeta(void);
  // PEFT related fields
  void *low_rank_activation;
  void *input_activation;
  std::unordered_map<PEFTModelID, LoraLinearModelState> model_state;
  size_t allocated_peft_buffer_size1 = 0, allocated_peft_buffer_size2 = 0;
};

namespace Kernels {
namespace LoraLinear {
void init_kernel_wrapper(LoraLinearMeta *m, int seed);
void inference_kernel_wrapper(LoraLinearMeta *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output);
void peft_bwd_kernel_wrapper(Context ctx,
                             Runtime *runtime,
                             LoraLinearMeta *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad);

namespace Internal {
template <typename DT>
void init_kernel(LoraLinearMeta *m, int seed, ffStream_t stream);
template <typename DT>
void inference_kernel(LoraLinearMeta *m,
                      BatchConfig const *bc,
                      DT const *input_ptr,
                      DT *output_ptr,
                      int in_dim,
                      int out_dim,
                      ffStream_t stream);
template <typename DT>
void peft_bwd_kernel(Context ctx,
                     Runtime *runtime,
                     LoraLinearMeta *m,
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
