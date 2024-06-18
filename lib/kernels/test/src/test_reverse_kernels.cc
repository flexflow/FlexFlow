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

    TensorShape shape = make_float_tensor_shape_w_legion_dims({num_elements});

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor = allocator.allocate_tensor(shape);
      GenericTensorAccessorW grad_input_accessor =
          create_filled_accessor_w(shape, allocator, 0.0f);

      Kernels::Reverse::forward_kernel(stream,
                                       input_accessor.get_float_ptr(),
                                       output_accessor.get_float_ptr(),
                                       num_out_blks,
                                       reverse_dim_size,
                                       in_blk_size,
                                       num_elements);

      SUBCASE("backward_kernel") {
        Kernels::Reverse::backward_kernel(stream,
                                          output_accessor.get_float_ptr(),
                                          grad_input_accessor.get_float_ptr(),
                                          num_out_blks,
                                          reverse_dim_size,
                                          in_blk_size,
                                          num_elements);

        std::vector<float> host_grad_input_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(grad_input_accessor));
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
