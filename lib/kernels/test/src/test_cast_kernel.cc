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

    TensorShape input_shape = make_float_tensor_shape_w_legion_dims({100, 100});
    TensorShape output_shape = get_double_tensor_shape({100, 100});

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR accessorR = read_only_accessor_from_write_accessor(
          create_random_filled_accessor_w(input_shape, allocator));
      GenericTensorAccessorW accessorW =
          allocator.allocate_tensor(output_shape);

      Kernels::Cast::forward_kernel(
          stream, accessorR, accessorW, DataType::FLOAT, DataType::DOUBLE);

      std::vector<double> host_double_data =
          load_data_to_host_from_device<double>(
              read_only_accessor_from_write_accessor(accessorW));

      for (size_t i = 0; i < host_double_data.size(); ++i) {
        REQUIRE(typeid(host_double_data[i]) == typeid(double));
      }

      SUBCASE("backward_kernel") {
        GenericTensorAccessorR grad_accessorR =
            read_only_accessor_from_write_accessor(
                create_random_filled_accessor_w(output_shape, allocator));
        GenericTensorAccessorW grad_accessorW =
            allocator.allocate_tensor(input_shape);

        Kernels::Cast::backward_kernel(stream,
                                       grad_accessorR,
                                       grad_accessorW,
                                       DataType::DOUBLE,
                                       DataType::FLOAT);

        std::vector<float> host_grad_float_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(grad_accessorW));

        for (size_t i = 0; i < host_grad_float_data.size(); ++i) {
          REQUIRE(typeid(host_grad_float_data[i]) == typeid(float));
        }
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
