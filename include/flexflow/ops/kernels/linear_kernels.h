#ifndef _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/linear.h"
#include <map>
#include <mutex>
#include <unordered_map>

namespace FlexFlow {

#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
struct cublasAlgoConfig_t {
  int batch_count;
  int m;
  int n;
  int k;
  int data_type;
  bool operator==(cublasAlgoConfig_t const &config) const {
    return (batch_count == config.batch_count) && (m == config.m) &&
           (n == config.n) && (k == config.k) &&
           (data_type == config.data_type);
  }
};

class cublasAlgoConfig_hasher {
public:
  std::size_t operator()(cublasAlgoConfig_t const &config) const {
    return config.batch_count * 98317ull ^ config.m * 49157ull ^
           config.n * 24593ull ^ config.k * 196613ull ^
           (config.data_type) * 6151ull;
  }
};
#endif

class LinearMeta : public OpMeta {
public:
  LinearMeta(FFHandler handle,
             int batch_size,
             Linear const *li,
             MemoryAllocator gpu_mem_allocator,
             int weightSize);
  ~LinearMeta(void);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  static std::mutex profile_lock;
  static std::unordered_map<cublasAlgoConfig_t, int, cublasAlgoConfig_hasher>
      algo_map;
  void findBestAlgoID(int m, int n, int k);
#else
  miopenTensorDescriptor_t outputTensor;
  miopenActivationDescriptor_t actiDesc;
#endif
  void *one_ptr;
  void *weight_ptr;
  DataType weight_ptr_type;
  DataType quantization_type;
  bool offload;
  char *quantized_weight_ptr;
  size_t quantized_weightSize;
  ActiMode activation;
  RegularizerMode kernel_reg_type;
  float kernel_reg_lambda;
  bool use_bias, add_bias_only_once;
  char op_name[MAX_OPNAME];
  Realm::RegionInstance reserveInst;
};

namespace Kernels {
namespace Linear {
void init_kernel(LinearMeta *m, int batch_size, int channel);
void forward_kernel_wrapper(LinearMeta const *m,
                            void const *input_ptr,
                            void *output_ptr,
                            void const *filter_ptr,
                            void const *bias_ptr,
                            int in_dim,
                            int out_dim,
                            int batch_size);
void backward_kernel_wrapper(LinearMeta const *m,
                             void const *input_ptr,
                             void *input_grad_ptr,
                             void const *output_ptr,
                             void *output_grad_ptr,
                             void const *kernel_ptr,
                             void *kernel_grad_ptr,
                             void *bias_ptr,
                             int in_dim,
                             int out_dim,
                             int batch_size);
bool use_activation(ActiMode mode);

namespace Internal {
template <typename DT>
void forward_kernel(LinearMeta const *m,
                    void const *input_ptr,
                    void *output_ptr,
                    void const *filter_ptr,
                    void const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size,
                    ffStream_t stream);
template <typename DT>
void backward_kernel(LinearMeta const *m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void *output_grad_ptr,
                     void const *kernel_ptr,
                     void *kernel_grad_ptr,
                     void *bias_ptr,
                     int in_dim,
                     int out_dim,
                     int batch_size,
                     ffStream_t stream);
template <typename DT>
__global__ void build_one_ptr(DT *one_ptr, int batch_size);
} // namespace Internal
} // namespace Linear
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
