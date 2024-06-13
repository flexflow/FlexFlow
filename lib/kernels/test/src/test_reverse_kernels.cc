#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/reverse_kernels.h"
#include "test_utils.h"
#include <algorithm>
#include <vector>

template <typename T>
void allocate_ptrs(std::vector<T **> &gpu_data_ptrs,
                   std::vector<size_t> const &num_elements,
                   Allocator &allocator) {
  for (size_t i = 0; i < gpu_data_ptrs.size(); ++i) {
    *gpu_data_ptrs[i] =
        static_cast<T *>(allocator.allocate(num_elements[i] * sizeof(float)));
  }
}

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reverse Forward and Backward Kernels") {
    std::size_t num_elements = 100;
    std::size_t reverse_dim_size = 10;
    std::size_t in_blk_size = 10;
    std::size_t num_out_blks = 1;

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    float *input_data, *output_data, *grad_input_data;
    std::vector<float **> ptrs = {&input_data, &output_data, &grad_input_data};
    std::vector<size_t> sizes = {num_elements, num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);

    fillDeviceDataNum(&input_data, num_elements, 1.0f);

    Kernels::Reverse::forward_kernel(stream,
                                     input_data,
                                     output_data,
                                     num_out_blks,
                                     reverse_dim_size,
                                     in_blk_size,
                                     num_elements);

    Kernels::Reverse::backward_kernel(stream,
                                      output_data,
                                      grad_input_data,
                                      num_out_blks,
                                      reverse_dim_size,
                                      in_blk_size,
                                      num_elements);

    std::vector<float> host_grad_input_data(num_elements);
    checkCUDA(cudaMemcpy(host_grad_input_data.data(),
                         grad_input_data,
                         num_elements * sizeof(float),
                         cudaMemcpyDeviceToHost));

    checkCUDA(cudaStreamDestroy(stream));
  }
}
