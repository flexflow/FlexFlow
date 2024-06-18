#include "doctest/doctest.h"
#include "kernels/concat_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test concat kernel forward and backward") {
    size_t num_inputs = 3;
    size_t size_per_input = 100;
    ff_dim_t concat_axis = ff_dim_t(0);

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    TensorShape input_shape =
        make_float_tensor_shape_w_legion_dims({size_per_input});

    Allocator allocator = get_local_memory_allocator();
    std::vector<GenericTensorAccessorR> input_accessors =
        repeat(num_inputs, [&]() {
          return read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(input_shape, allocator));
        });

    GenericTensorAccessorW output_accessor =
        allocator.allocate_tensor(input_shape);

    SUBCASE("forward_kernel") {
      Kernels::Concat::forward_kernel(
          stream, output_accessor, input_accessors, concat_axis);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      for (int i = 0; i < num_inputs; i++) {
        std::vector<float> input_data =
            load_data_to_host_from_device<float>(input_accessors[i]);
        auto output_start = host_output_data.begin() + i * size_per_input;
        REQUIRE(std::equal(
            output_start, output_start + size_per_input, input_data.begin()));
      }

      SUBCASE("backward_kernel") {
        std::vector<GenericTensorAccessorW> grad_input_accessors;
        for (int i = 0; i < num_inputs; i++) {
          grad_input_accessors.push_back(
              allocator.allocate_tensor(input_shape));
          fill_tensor_accessor_w(grad_input_accessors[i], 0.0f);
        }

        void *grad_output_data_ptr =
            allocator.allocate(num_inputs * size_per_input * sizeof(float));
        checkCUDA(cudaMemcpy(grad_output_data_ptr,
                             host_output_data.data(),
                             host_output_data.size() * sizeof(float),
                             cudaMemcpyHostToDevice));

        GenericTensorAccessorR grad_output_accessor{
            DataType::FLOAT, input_shape, grad_output_data_ptr};        

        Kernels::Concat::backward_kernel(
            stream, grad_output_accessor, grad_input_accessors, concat_axis);

        for (int i = 0; i < num_inputs; i++) {
          std::vector<float> host_grad_input =
              load_data_to_host_from_device<float>(
                  read_only_accessor_from_write_accessor(
                      grad_input_accessors[i]));
          auto grad_output_start =
              host_output_data.begin() + i * size_per_input;
          REQUIRE(std::equal(host_grad_input.begin(),
                             host_grad_input.end(),
                             grad_output_start));
        }
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
