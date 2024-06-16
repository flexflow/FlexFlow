#include "doctest/doctest.h"
#include "kernels/split_kernels.h"
#include "test_utils.h"
#include <numeric>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Split Forward and Backward Kernel") {
    int const num_elements = 100;
    int const num_outputs = 2;
    coord_t out_blk_sizes[] = {50, 50};
    const coord_t in_blk_size = 100;
    const coord_t num_blks = 1;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Allocator allocator = get_local_memory_allocator();

    TensorShape input_shape = get_float_tensor_shape({num_elements});
    GenericTensorAccessorW input_accessor =
        getRandomFilledAccessorW(input_shape,
                                 allocator);
    std::vector<float> host_input_data =
        fill_host_data<float>(input_accessor.ptr, num_elements);                              

    std::vector<float *> output_ptrs(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
      output_ptrs[i] = static_cast<float *>(
          allocator.allocate(out_blk_sizes[i] * sizeof(float)));
    }

    SUBCASE("Test Split Forward Kernel") {
      Kernels::Split::forward_kernel(stream,
                                     output_ptrs.data(),
                                     (const float*)input_accessor.ptr,
                                     out_blk_sizes,
                                     in_blk_size,
                                     num_blks,
                                     num_outputs);

      std::vector<std::vector<float>> host_output_data(
          num_outputs, std::vector<float>(50, 0));
      for (int i = 0; i < num_outputs; i++) {
        cudaMemcpy(host_output_data[i].data(),
                   output_ptrs[i],
                   out_blk_sizes[i] * sizeof(float),
                   cudaMemcpyDeviceToHost);
      }

      for (int i = 0; i < num_outputs; i++) {
        int offset = std::accumulate(out_blk_sizes, out_blk_sizes + i, 0);
        for (int j = 0; j < out_blk_sizes[i]; j++) {
          REQUIRE(host_output_data[i][j] == host_input_data[offset + j]);
        }
      }
    }

    SUBCASE("Test Split Backward Kernel") {
      float *grad_input_data = static_cast<float *>(
          allocator.allocate(num_elements * sizeof(float)));
      cudaMemset(grad_input_data, 0, num_elements * sizeof(float));

      Kernels::Split::backward_kernel(
          stream,
          grad_input_data,
          const_cast<float const **>(output_ptrs.data()),
          out_blk_sizes,
          in_blk_size,
          num_blks,
          num_outputs);

      std::vector<float> host_grad_input_data(num_elements, 0);
      cudaMemcpy(host_grad_input_data.data(),
                 grad_input_data,
                 num_elements * sizeof(float),
                 cudaMemcpyDeviceToHost);
    }

    cudaStreamDestroy(stream);
  }
}
