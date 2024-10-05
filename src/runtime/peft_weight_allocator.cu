

#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/utils/peft_weight_allocator.h"
#include "flexflow/utils/cuda_helper.h"
#include <random>
#include <vector>
namespace FlexFlow {

template <typename DT>
void init_kernel(LoraLinearWeight const &weight, int in_dim, int out_dim, int rank, int seed, cudaStream_t stream) {
    // Initialize generator
    std::mt19937 gen(seed);

    // Get handle to weights by iterating over m->model_state to get each
    // LoraLinearWeight object
    int w0_num_elements = rank * in_dim;
    int w1_num_elements = rank * out_dim;

    // LoRA_A weight: [in_dim, rank]
    float stdv_lora_a = 1.0f / sqrt(in_dim);
    std::uniform_real_distribution<float> dis_lora_a(-stdv_lora_a, stdv_lora_a);
    std::vector<DT> lora_a_random_init(w0_num_elements);
    for (auto &num : lora_a_random_init) {
        float num_float = dis_lora_a(gen);
        if (std::is_same<DT, half>::value) {
            num = __float2half(num_float);
        } else {
            num = num_float;
        }
    }
    checkCUDA(cudaMemcpyAsync(static_cast<DT *>(weight.w0_ptr),
                                lora_a_random_init.data(),
                                w0_num_elements * sizeof(DT),
                                cudaMemcpyHostToDevice,
                                stream));

    // LoRA_B weight: [rank, out_dim]
    float stdv_lora_b = 1.0f / sqrt(rank);
    std::uniform_real_distribution<float> dis_lora_b(-stdv_lora_b, stdv_lora_b);
    std::vector<float> lora_b_random_init(w1_num_elements);
    for (auto &num : lora_b_random_init) {
        float num_float = dis_lora_b(gen);
        if (std::is_same<DT, half>::value) {
            num = __float2half(num_float);
        } else {
            num = num_float;
        }
    }
    checkCUDA(cudaMemcpyAsync(static_cast<DT *>(w1_ptr),
                                lora_b_random_init.data(),
                                w1_num_elements * sizeof(DT),
                                cudaMemcpyHostToDevice,
                                stream));
}

void init_peft_weight_wrapper(LoraLinearWeight const &weight, int in_dim, int out_dim, int rank, DataType dt, int seed) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  if (dt == DT_FLOAT) {
    Internal::init_kernel<float>(weight, in_di, out_dim, rank, seed, stream);
  } else if (dt == DT_HALF) {
    Internal::init_kernel<half>(weight, in_di, out_dim, rank, seed, stream);
  } else {
    assert(false && "Unsupported data type");
  }
}

} // namespace FlexFlow