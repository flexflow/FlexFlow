#include "doctest/doctest.h"
#include "kernels/combine_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test combine kernel") {
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    Allocator allocator = get_local_memory_allocator();

    TensorShape input_shape = make_float_tensor_shape_w_legion_dims({100, 100});

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR accessorR = read_only_accessor_from_write_accessor(
          create_random_filled_accessor_w(input_shape, allocator));
      GenericTensorAccessorW accessorW = allocator.allocate_tensor(input_shape);

      Kernels::Combine::forward_kernel(stream, accessorR, accessorW);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(accessorW));
      REQUIRE(contains_non_zero(host_output_data));

      SUBCASE("backward_kernel") {
        GenericTensorAccessorR accessorRGrad =
            read_only_accessor_from_write_accessor(
                allocator.allocate_tensor(input_shape));
        GenericTensorAccessorW accessorWGrad =
            allocator.allocate_tensor(input_shape);

        Kernels::Combine::backward_kernel(stream, accessorRGrad, accessorWGrad);

        std::vector<float> host_input_grad =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(accessorWGrad));
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
