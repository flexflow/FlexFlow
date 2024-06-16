#include "doctest/doctest.h"
#include "kernels/combine_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test combine kernel") {
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    Allocator allocator = get_local_memory_allocator();

    TensorShape input_shape = get_float_tensor_shape({100, 100});

    SUBCASE("Test combine kernel forward") {
      GenericTensorAccessorR accessorR = makeReadOnlyAccessor(
          getRandomFilledAccessorW(input_shape, allocator));
      GenericTensorAccessorW accessorW = allocator.allocate_tensor(input_shape);

      Kernels::Combine::forward_kernel(stream, accessorR, accessorW);

      std::vector<float> host_output_data =
          fill_host_data<float>(accessorW.ptr, 100 * 100);
      REQUIRE(contains_non_zero(host_output_data));

      SUBCASE("Test combine kernel backward") {
        GenericTensorAccessorR accessorRGrad =
            makeReadOnlyAccessor(allocator.allocate_tensor(input_shape));
        GenericTensorAccessorW accessorWGrad =
            allocator.allocate_tensor(input_shape);

        Kernels::Combine::backward_kernel(stream, accessorRGrad, accessorWGrad);

        std::vector<float> host_input_grad =
            fill_host_data<float>(accessorWGrad.ptr, 100 * 100);
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
