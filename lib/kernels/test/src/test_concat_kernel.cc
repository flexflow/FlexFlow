#include "doctest/doctest.h"
#include "kernels/concat_kernels.h"
#include "test_utils.h"

namespace FlexFlow {
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test concat kernel forward and backward") {
    int const num_inputs = 3;
    int const size_per_input = 100;
    ff_dim_t concat_axis = ff_dim_t(0);

    ArrayShape shape = ArrayShape{
        std::vector<size_t>{size_per_input},
    };

    Allocator allocator = get_local_memory_allocator();
    std::vector<void *> input_ptrs;
    std::vector<GenericTensorAccessorR> input_accessors;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < num_inputs; i++) {
      void *input_data_ptr = allocator.allocate(size_per_input * sizeof(float));
      input_ptrs.push_back(input_data_ptr);
      GenericTensorAccessorR accessor{DataType::FLOAT, shape, input_data_ptr};
      input_accessors.push_back(accessor);

      std::vector<float> host_input_data(size_per_input);
      for (float &val : host_input_data) {
        val = dist(gen);
      }
      checkCUDA(cudaMemcpy(input_data_ptr,
                           host_input_data.data(),
                           host_input_data.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
    }

    void *output_data_ptr =
        allocator.allocate(num_inputs * size_per_input * sizeof(float));
    const GenericTensorAccessorW output_accessor{
        DataType::FLOAT, shape, output_data_ptr};

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Kernels::Concat::forward_kernel(
        stream, output_accessor, input_accessors, concat_axis);

    std::vector<float> host_output_data(num_inputs * size_per_input);
    checkCUDA(cudaMemcpy(host_output_data.data(),
                         output_data_ptr,
                         host_output_data.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_inputs; i++) {
      std::vector<float> temp(size_per_input);
      checkCUDA(cudaMemcpy(temp.data(),
                           input_ptrs[i],
                           size_per_input * sizeof(float),
                           cudaMemcpyDeviceToHost));
      for (int j = 0; j < size_per_input; j++) {
        REQUIRE(host_output_data[i * size_per_input + j] == temp[j]);
      }
    }

    std::vector<void *> grad_input_ptrs;
    std::vector<GenericTensorAccessorW> grad_input_accessors;
    for (int i = 0; i < num_inputs; i++) {
      void *grad_input_data_ptr =
          allocator.allocate(size_per_input * sizeof(float));
      grad_input_ptrs.push_back(grad_input_data_ptr);
      GenericTensorAccessorW accessor{
          DataType::FLOAT, shape, grad_input_data_ptr};
      grad_input_accessors.push_back(accessor);
      cudaMemset(grad_input_data_ptr, 0, size_per_input * sizeof(float));
    }

    void *grad_output_data_ptr =
        allocator.allocate(num_inputs * size_per_input * sizeof(float));
    checkCUDA(cudaMemcpy(grad_output_data_ptr,
                         host_output_data.data(),
                         host_output_data.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    const GenericTensorAccessorR grad_output_accessor{
        DataType::FLOAT, shape, grad_output_data_ptr};

    Kernels::Concat::backward_kernel(
        stream, grad_output_accessor, grad_input_accessors, concat_axis);

    for (int i = 0; i < num_inputs; i++) {
      std::vector<float> host_grad_input(size_per_input);
      checkCUDA(cudaMemcpy(host_grad_input.data(),
                           grad_input_ptrs[i],
                           size_per_input * sizeof(float),
                           cudaMemcpyDeviceToHost));
      for (int j = 0; j < size_per_input; j++) {
        REQUIRE(host_grad_input[j] == host_output_data[i * size_per_input + j]);
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
} // namespace FlexFlow
