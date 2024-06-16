#include "doctest/doctest.h"
#include "kernels/transpose_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Transpose Kernel Operations") {
    std::size_t num_elements = 100;
    std::size_t num_dims = 2;
    TensorShape shape = get_float_tensor_shape({10, 10});

    std::vector<ff_dim_t> perm = {ff_dim_t(0), ff_dim_t(1)};

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    TransposePerDeviceState state =
        Kernels::Transpose::init_kernel(num_dims, perm);

    SUBCASE("Test Transpose Forward Kernel") {
      GenericTensorAccessorR input_accessor =
          makeReadOnlyAccessor(getRandomFilledAccessorW(shape, allocator));
      GenericTensorAccessorW output_accessor = allocator.allocate_tensor(shape);

      Kernels::Transpose::forward_kernel(
          stream, state, input_accessor, output_accessor);

      std::vector<float> host_output_data =
          fill_host_data<float>(output_accessor.ptr, num_elements);

      SUBCASE("Test Transpose Backward Kernel") {
        GenericTensorAccessorW input_grad_accessor =
            getRandomFilledAccessorW(shape, allocator);

        GenericTensorAccessorR output_grad_accessor =
            makeReadOnlyAccessor(allocator.allocate_tensor(shape));

        Kernels::Transpose::backward_kernel(
            stream, state, input_grad_accessor, output_grad_accessor);

        std::vector<float> host_grad_input_data =
            fill_host_data<float>(input_grad_accessor.ptr, num_elements);
      }
    }

    cleanup_test(stream, handle);
  }
}
