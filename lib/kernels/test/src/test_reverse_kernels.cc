#include "doctest/doctest.h"
#include "kernels/reverse_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reverse Forward and Backward Kernels") {
    std::size_t num_elements = 100;
    std::size_t reverse_dim_size = 10;
    std::size_t in_blk_size = 10;
    std::size_t num_out_blks = 1;

    TensorShape shape = get_float_tensor_shape({num_elements});

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    SUBCASE("Test Reverse Kernel Forward") {
      GenericTensorAccessorR input_accessor =
          makeReadOnlyAccessor(getFilledAccessorW(shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor = allocator.allocate_tensor(shape);
      GenericTensorAccessorW grad_input_accessor =
          getFilledAccessorW(shape, allocator, 0.0f);

      Kernels::Reverse::forward_kernel(stream,
                                       (float const *)input_accessor.ptr,
                                       (float *)output_accessor.ptr,
                                       num_out_blks,
                                       reverse_dim_size,
                                       in_blk_size,
                                       num_elements);

      SUBCASE("Test Reverse Kernel Backward") {
        Kernels::Reverse::backward_kernel(stream,
                                          (float const *)output_accessor.ptr,
                                          (float *)grad_input_accessor.ptr,
                                          num_out_blks,
                                          reverse_dim_size,
                                          in_blk_size,
                                          num_elements);

        std::vector<float> host_grad_input_data =
            fill_host_data<float>(grad_input_accessor.ptr, num_elements);
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
