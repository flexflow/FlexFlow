#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/reshape_kernels.h"
#include <vector>
#include <algorithm> 
#include <iostream>

namespace FlexFlow {
TEST_SUITE(FF_TEST_SUITE) {
    TEST_CASE("Test Reshape Forward and Backward") {
        std::size_t num_elements = 100;
        std::size_t dims[] = {num_elements};
        std::size_t num_dims = 1;
        FlexFlow::ArrayShape shape(dims, num_dims);

        Allocator allocator = get_local_memory_allocator();
        ReshapePerDeviceState state = Kernels::Reshape::init_kernel(DataType::FLOAT);

        float* input_data = static_cast<float*>(allocator.allocate(num_elements * sizeof(float)));
        const GenericTensorAccessorR input_accessor{DataType::FLOAT, shape, input_data};
        std::vector<float> host_input_data(num_elements, 1.0f);
        checkCUDA(cudaMemcpy(input_data, host_input_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

        float* output_data = static_cast<float*>(allocator.allocate(num_elements * sizeof(float)));
        const GenericTensorAccessorW forward_output_accessor{DataType::FLOAT, shape, output_data};
        std::vector<float> check_output_data(num_elements);

        cudaStream_t stream;
        checkCUDA(cudaStreamCreate(&stream));

        Kernels::Reshape::forward_kernel(stream, state, input_accessor, forward_output_accessor);

        checkCUDA(cudaMemcpy(check_output_data.data(), output_data, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

        for (std::size_t i = 0; i < num_elements; ++i) {
            REQUIRE(host_input_data[i] == check_output_data[i]);
        }

        std::vector<float> host_grad_output_data(num_elements, 1.0f);
        float* grad_data = static_cast<float*>(allocator.allocate(num_elements * sizeof(float)));
        checkCUDA(cudaMemcpy(grad_data, host_grad_output_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
        const GenericTensorAccessorR grad_accessor{DataType::FLOAT, shape, grad_data};

        Kernels::Reshape::backward_kernel(stream, state, forward_output_accessor, grad_accessor);

        std::vector<float> host_grad_input_data(num_elements);
        checkCUDA(cudaMemcpy(host_grad_input_data.data(), output_data, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

        for (std::size_t i = 0; i < num_elements; ++i) {
            CHECK(host_grad_input_data[i] == 2.0f);
        }
        checkCUDA(cudaStreamDestroy(stream));
    }
}
} // namespace FlexFlow
