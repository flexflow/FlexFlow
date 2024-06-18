#include "doctest/doctest.h"
#include "kernels/flat_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Flat Kernel") {
    std::size_t num_elements = 100;

    TensorShape input_shape =
        make_float_tensor_shape_w_legion_dims({num_elements});

    Allocator allocator = get_local_memory_allocator();

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    GenericTensorAccessorR input_accessor =
        read_only_accessor_from_write_accessor(
            create_filled_accessor_w(input_shape, allocator, 2.0f));
    GenericTensorAccessorW output_accessor =
        allocator.allocate_tensor(input_shape);

    SUBCASE("forward_kernel") {
      Kernels::Flat::forward_kernel(
          stream, input_accessor, output_accessor.get_float_ptr());

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(2.0f == check_output_data[i]);
      }
      SUBCASE("backward_kernel") {
        GenericTensorAccessorR data_accessor =
            read_only_accessor_from_write_accessor(
                create_filled_accessor_w(input_shape, allocator, 1.0f));

        Kernels::Flat::backward_kernel(stream,
                                       input_accessor,
                                       output_accessor.get_float_ptr(),
                                       data_accessor.get_float_ptr());

        std::vector<float> backward_output_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(output_accessor));

        bool correct_output = std::all_of(backward_output_data.begin(),
                                          backward_output_data.end(),
                                          [](float x) { return x == 3.0f; });

        CHECK(correct_output);
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
