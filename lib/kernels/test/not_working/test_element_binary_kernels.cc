// #include "doctest/doctest.h"
// #include "kernels/local_allocator.h"
// #include "kernels/element_binary_kernels.h"
// #include <vector>
// #include <algorithm>
// #include <cmath>

// using namespace ::FlexFlow;
// TEST_SUITE(FF_TEST_SUITE) {
//     TEST_CASE("Test Element Binary Forward and Backward Kernel") {
//         std::size_t num_elements = 100;
//         std::size_t dims[] = {10, 10};
//         ArrayShape shape(dims, 2);

//         OperatorType op_type = OperatorType::EW_ADD;  // Example operation
//         bool should_broadcast_lhs = false;
//         bool should_broadcast_rhs = false;

//         PerDeviceFFHandle handle;
//         cudnnCreate(&handle.dnn);
//         cublasCreate(&handle.blas);
//         handle.workSpaceSize = 1024 * 1024;
//         cudaMalloc(&handle.workSpace, handle.workSpaceSize);
//         handle.allowTensorOpMathConversion = true;

//         Allocator allocator = get_local_memory_allocator();
//         ElementBinaryPerDeviceState state =
//         Kernels::ElementBinary::init_kernel(handle, op_type,
//         should_broadcast_lhs, should_broadcast_rhs, shape, shape, shape);

//         float* lhs_data = static_cast<float*>(allocator.allocate(num_elements
//         * sizeof(float))); float* rhs_data =
//         static_cast<float*>(allocator.allocate(num_elements *
//         sizeof(float))); float* output_data =
//         static_cast<float*>(allocator.allocate(num_elements *
//         sizeof(float)));

//         std::vector<float> host_lhs_data(num_elements);
//         std::vector<float> host_rhs_data(num_elements);
//         std::generate(host_lhs_data.begin(), host_lhs_data.end(), []() {
//         return static_cast<float>(rand()) / RAND_MAX; });
//         std::generate(host_rhs_data.begin(), host_rhs_data.end(), []() {
//         return static_cast<float>(rand()) / RAND_MAX; });
//         checkCUDA(cudaMemcpy(lhs_data, host_lhs_data.data(), num_elements *
//         sizeof(float), cudaMemcpyHostToDevice));
//         checkCUDA(cudaMemcpy(rhs_data, host_rhs_data.data(), num_elements *
//         sizeof(float), cudaMemcpyHostToDevice));

//         cudaStream_t stream;
//         checkCUDA(cudaStreamCreate(&stream));

//         // Forward pass
//         Kernels::ElementBinary::forward_kernel(stream, state, lhs_data,
//         rhs_data, output_data, op_type, should_broadcast_lhs, handle);

//         std::vector<float> host_output_data(num_elements);
//         checkCUDA(cudaMemcpy(host_output_data.data(), output_data,
//         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

//         // Verify output of forward pass
//         for (int i = 0; i < num_elements; ++i) {
//             float expected_value = host_lhs_data[i] + host_rhs_data[i];
//             CHECK(doctest::Approx(host_output_data[i]) == expected_value);
//         }

//         // Setup for backward pass
//         float* grad_output_data =
//         static_cast<float*>(allocator.allocate(num_elements *
//         sizeof(float))); std::vector<float>
//         host_grad_output_data(num_elements, 1.0f);  // Assuming gradient from
//         checkCUDA(cudaMemcpy(grad_output_data,
//         host_grad_output_data.data(), num_elements * sizeof(float),
//         cudaMemcpyHostToDevice));

//         float* lhs_grad_data =
//         static_cast<float*>(allocator.allocate(num_elements *
//         sizeof(float))); float* rhs_grad_data =
//         static_cast<float*>(allocator.allocate(num_elements *
//         sizeof(float)));

//         // Backward pass
//         Kernels::ElementBinary::backward_kernel(stream, state,
//         grad_output_data, lhs_data, rhs_data, lhs_grad_data, rhs_grad_data,
//         op_type, should_broadcast_lhs, should_broadcast_rhs, handle);

//         std::vector<float> host_lhs_grad_data(num_elements);
//         std::vector<float> host_rhs_grad_data(num_elements);
//         checkCUDA(cudaMemcpy(host_lhs_grad_data.data(), lhs_grad_data,
//         num_elements * sizeof(float), cudaMemcpyDeviceToHost));
//         checkCUDA(cudaMemcpy(host_rhs_grad_data.data(), rhs_grad_data,
//         num_elements * sizeof(float), cudaMemcpyDeviceToHost));


//         checkCUDA(cudaStreamDestroy(stream));
//     }
// }
