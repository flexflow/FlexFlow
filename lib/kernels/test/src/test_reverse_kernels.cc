#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/reverse_kernels.h"
#include <algorithm>
#include <iostream>
#include <vector>

namespace FlexFlow {

TEST_SUITE("ReverseKernelTests") {
  TEST_CASE("Test Reverse Forward and Backward Kernels") {
    std::size_t num_elements = 100;
    std::size_t reverse_dim_size = 10;
    std::size_t in_blk_size = 10;
    std::size_t num_out_blks = 1;

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();
    float *input_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    float *output_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    float *grad_input_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));

    std::vector<float> host_input_data(num_elements);
    std::iota(host_input_data.begin(), host_input_data.end(), 0.0f);
    checkCUDA(cudaMemcpy(input_data, host_input_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));

    Kernels::Reverse::forward_kernel(stream, input_data, output_data,
                                     num_out_blks, reverse_dim_size,
                                     in_blk_size, num_elements);

    std::vector<float> host_grad_output_data(num_elements, 1.0f);
    checkCUDA(cudaMemcpy(output_data, host_grad_output_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));

    Kernels::Reverse::backward_kernel(stream, output_data, grad_input_data,
                                      num_out_blks, reverse_dim_size,
                                      in_blk_size, num_elements);

    std::vector<float> host_grad_input_data(num_elements);
    checkCUDA(cudaMemcpy(host_grad_input_data.data(), grad_input_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_elements; i++) {
      CHECK(doctest::Approx(host_grad_input_data[i]) == 1.0f);
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}

} // namespace FlexFlow
