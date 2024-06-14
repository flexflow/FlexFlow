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

    TensorShape input_shape_a = TensorShape{
        TensorDims{
            FFOrdered<size_t>{m, k, batch},
        },
        DataType::FLOAT,
    };

    TensorShape input_shape_b = TensorShape{
        TensorDims{
            FFOrdered<size_t>{k, n, batch},
        },
        DataType::FLOAT,
    };

    TensorShape output_shape = TensorShape{
        TensorDims{
            FFOrdered<size_t>{m, n, batch},
        },
        DataType::FLOAT,
    };

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    GenericTensorAccessorW accessor_a =
        allocator.allocate_tensor(input_shape_a);
    GenericTensorAccessorW accessor_b =
        allocator.allocate_tensor(input_shape_b);
    GenericTensorAccessorW accessor_output =
        allocator.allocate_tensor(output_shape);

    SUBCASE("Test BatchMatmul Forward") {
      Kernels::BatchMatmul::forward_kernel(stream,
                                           handle,
                                           (float *)accessor_output.ptr,
                                           (float *)accessor_a.ptr,
                                           (float *)accessor_b.ptr,
                                           m,
                                           n,
                                           k,
                                           batch,
                                           a_seq_length_dim,
                                           b_seq_length_dim,
                                           seq_length);
    }

    SUBCASE("Test BatchMatmul Backward") {
      GenericTensorAccessorW a_grad_accessor =
          allocator.allocate_tensor(input_shape_a);
      GenericTensorAccessorW b_grad_accessor =
          allocator.allocate_tensor(input_shape_b);
      GenericTensorAccessorW o_grad_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::BatchMatmul::backward_kernel(stream,
                                            handle,
                                            (float *)accessor_output.ptr,
                                            (float *)o_grad_accessor.ptr,
                                            (float *)accessor_a.ptr,
                                            (float *)a_grad_accessor.ptr,
                                            (float *)accessor_b.ptr,
                                            (float *)b_grad_accessor.ptr,
                                            m,
                                            n,
                                            k,
                                            batch);
    }

    cudaStreamDestroy(stream);
  }
}
