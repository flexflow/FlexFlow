#include "doctest/doctest.h"
#include "kernels/batch_matmul_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test BatchMatmul Kernel") {
    size_t m = 10;
    size_t n = 10;
    size_t k = 10;
    size_t batch = 5;
    size_t a_seq_length_dim = -1;
    size_t b_seq_length_dim = -1;
    size_t seq_length = -1;

    PerDeviceFFHandle handle = get_per_device_ff_handle();
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    TensorShape input_shape_a =
        make_float_tensor_shape_w_legion_dims({m, k, batch});
    TensorShape input_shape_b =
        make_float_tensor_shape_w_legion_dims({k, n, batch});
    TensorShape output_shape =
        make_float_tensor_shape_w_legion_dims({m, n, batch});

    GenericTensorAccessorW accessor_a =
        create_random_filled_accessor_w(input_shape_a, allocator);
    GenericTensorAccessorW accessor_b =
        create_random_filled_accessor_w(input_shape_b, allocator);
    GenericTensorAccessorW accessor_output =
        allocator.allocate_tensor(output_shape);

    SUBCASE("forward_kernel") {
      Kernels::BatchMatmul::forward_kernel(stream,
                                           handle,
                                           accessor_output.get_float_ptr(),
                                           accessor_a.get_float_ptr(),
                                           accessor_b.get_float_ptr(),
                                           m,
                                           n,
                                           k,
                                           batch,
                                           a_seq_length_dim,
                                           b_seq_length_dim,
                                           seq_length);
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW o_grad_accessor =
          create_random_filled_accessor_w(output_shape, allocator);
      GenericTensorAccessorW a_grad_accessor =
          allocator.allocate_tensor(input_shape_a);
      GenericTensorAccessorW b_grad_accessor =
          allocator.allocate_tensor(input_shape_b);

      Kernels::BatchMatmul::backward_kernel(stream,
                                            handle,
                                            accessor_output.get_float_ptr(),
                                            o_grad_accessor.get_float_ptr(),
                                            accessor_a.get_float_ptr(),
                                            a_grad_accessor.get_float_ptr(),
                                            accessor_b.get_float_ptr(),
                                            b_grad_accessor.get_float_ptr(),
                                            m,
                                            n,
                                            k,
                                            batch);
    }

    cleanup_test(stream, handle);
  }
}
