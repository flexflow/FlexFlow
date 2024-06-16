#include "doctest/doctest.h"
#include "kernels/cast_kernels.h"
#include "test_utils.h"
#include <type_traits>

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test cast kernel") {
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    TensorShape input_shape = get_float_tensor_shape({100, 100});
    TensorShape output_shape = get_double_tensor_shape({100, 100});

    SUBCASE("Test forward cast kernel") {
      GenericTensorAccessorR accessorR = makeReadOnlyAccessor(
          getRandomFilledAccessorW(input_shape, allocator));
      GenericTensorAccessorW accessorW =
          allocator.allocate_tensor(output_shape);

      Kernels::Cast::forward_kernel(
          nullptr, accessorR, accessorW, DataType::FLOAT, DataType::DOUBLE);

      std::vector<double> host_double_data =
          fill_host_data<double>(accessorW.ptr, 100 * 100);

      for (size_t i = 0; i < host_double_data.size(); ++i) {
        REQUIRE(typeid(host_double_data[i]) == typeid(double));
      }

      SUBCASE("Test backward cast kernel") {
        GenericTensorAccessorR grad_accessorR = makeReadOnlyAccessor(
            getRandomFilledAccessorW(output_shape, allocator));
        GenericTensorAccessorW grad_accessorW =
            allocator.allocate_tensor(input_shape);

        Kernels::Cast::backward_kernel(nullptr,
                                       grad_accessorR,
                                       grad_accessorW,
                                       DataType::DOUBLE,
                                       DataType::FLOAT);

        std::vector<float> host_grad_float_data =
            fill_host_data<float>(grad_accessorW.ptr, 100 * 100);

        for (size_t i = 0; i < host_grad_float_data.size(); ++i) {
          REQUIRE(typeid(host_grad_float_data[i]) == typeid(float));
        }
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
