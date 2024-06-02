// #include "doctest/doctest.h"
// #include "kernels/local_allocator.h"
// #include "kernels/reduce_kernels.h"
// #include <algorithm>
// #include <iostream>
// #include <vector>

// using namespace ::FlexFlow;

// TEST_SUITE(FF_TEST_SUITE) {
//   TEST_CASE("Test Reduce Forward and Backward Kernel") {
//     std::size_t num_elements = 100;
//     std::size_t output_elements = 10;
//     std::size_t dims[] = {10, 10};
//     std::size_t output_dims[] = {10, 1};
//     ArrayShape input_shape(dims, 2);
//     ArrayShape output_shape(output_dims, 2);
//     OperatorType op_type = OperatorType::REDUCE_SUM;
//     size_t reduction_size = 10;

//     PerDeviceFFHandle handle;
//     cudnnCreate(&handle.dnn);
//     cublasCreate(&handle.blas);
//     handle.workSpaceSize = 1024 * 1024;
//     cudaMalloc(&handle.workSpace, handle.workSpaceSize);
//     handle.allowTensorOpMathConversion = true;

//     Allocator allocator = get_local_memory_allocator();
//     ReducePerDeviceState state = Kernels::Reduce::init_kernel(
//         handle, op_type, reduction_size, input_shape, output_shape);

//     float *input_data =
//         static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
//     float *output_data = static_cast<float *>(
//         allocator.allocate(output_elements * sizeof(float)));

//     std::vector<float> host_input_data(num_elements);
//     std::generate(host_input_data.begin(), host_input_data.end(),
//                   []() { return static_cast<float>(rand()) / RAND_MAX; });
//     checkCUDA(cudaMemcpy(input_data, host_input_data.data(),
//                          num_elements * sizeof(float), cudaMemcpyHostToDevice));

//     cudaStream_t stream;
//     checkCUDA(cudaStreamCreate(&stream));

//     Kernels::Reduce::forward_kernel(stream, state, input_data, output_data);

//     float *grad_input_data =
//         static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
//     std::fill_n(host_input_data.begin(), num_elements, 1.0f);
//     checkCUDA(cudaMemcpy(grad_input_data, host_input_data.data(),
//                          num_elements * sizeof(float), cudaMemcpyHostToDevice));

//     Kernels::Reduce::backward_kernel(stream, state, output_data,
//                                      grad_input_data);

//     std::vector<float> host_grad_input_data(num_elements);
//     checkCUDA(cudaMemcpy(host_grad_input_data.data(), grad_input_data,
//                          num_elements * sizeof(float), cudaMemcpyDeviceToHost));

//     checkCUDA(cudaStreamDestroy(stream));
//   }
// }
