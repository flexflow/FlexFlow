#include "doctest/doctest.h"
#include "kernels/combine_kernels.h"
#include "kernels/local_allocator.h"

#include <random>

namespace FlexFlow {
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test combine kernel forward") {
    std::size_t dims[] = {100, 100};
    std::size_t num_dims = 2;
    FlexFlow::ArrayShape shape(dims, num_dims);

    Allocator allocator = get_local_memory_allocator();
    void *input_data_ptr = allocator.allocate(100 * 100 * sizeof(float));
    void *output_data_ptr = allocator.allocate(100 * 100 * sizeof(float));

    const GenericTensorAccessorR accessorR{DataType::FLOAT, shape,
                                           input_data_ptr};
    const GenericTensorAccessorW accessorW{DataType::FLOAT, shape,
                                           output_data_ptr};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> host_input_data(100 * 100);
    for (auto &val : host_input_data) {
      val = dist(gen);
    }

    checkCUDA(cudaMemcpy(input_data_ptr, host_input_data.data(),
                         host_input_data.size() * sizeof(float),
                         cudaMemcpyHostToDevice));

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Kernels::Combine::forward_kernel(stream, accessorR, accessorW);

    std::vector<float> host_output_data(100 * 100);
    checkCUDA(cudaMemcpy(host_output_data.data(), output_data_ptr,
                         host_output_data.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < host_input_data.size(); ++i) {
      REQUIRE(host_output_data[i] == host_input_data[i]);
    }

    checkCUDA(cudaStreamDestroy(stream));
  }

  TEST_CASE("Test combine kernel backward") {
    std::size_t dims[] = {100, 100};
    std::size_t num_dims = 2;
    FlexFlow::ArrayShape shape(dims, num_dims);

    Allocator allocator = get_local_memory_allocator();
    void *grad_output_data_ptr = allocator.allocate(100 * 100 * sizeof(float));
    void *grad_input_data_ptr = allocator.allocate(100 * 100 * sizeof(float));

    std::vector<float> host_output_grad(100 * 100, 1.0f);
    std::vector<float> host_input_grad(100 * 100, 0.0f);

    checkCUDA(cudaMemcpy(grad_output_data_ptr, host_output_grad.data(),
                         host_output_grad.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(grad_input_data_ptr, host_input_grad.data(),
                         host_input_grad.size() * sizeof(float),
                         cudaMemcpyHostToDevice));

    const GenericTensorAccessorR accessorRGrad{DataType::FLOAT, shape,
                                               grad_output_data_ptr};
    const GenericTensorAccessorW accessorWGrad{DataType::FLOAT, shape,
                                               grad_input_data_ptr};

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Kernels::Combine::backward_kernel(stream, accessorRGrad, accessorWGrad);

    checkCUDA(cudaMemcpy(host_input_grad.data(), grad_input_data_ptr,
                         host_input_grad.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (float val : host_input_grad) {
      REQUIRE(val == 1.0f);
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
} // namespace FlexFlow
