// #include "doctest/doctest.h"
// #include "kernels/linear_kernels.h"
// #include "kernels/local_allocator.h"
// #include <algorithm>
// #include <iostream>
// #include <vector>

// using namespace ::FlexFlow;
// TEST_SUITE(FF_TEST_SUITE) {
//   TEST_CASE("Test Linear Forward and Backward Kernel") {
//     std::cout << "Test Linear Forward and Backward Kernel" << std::endl;
//     int batch_size = 10;
//     int in_dim = 5;
//     int out_dim = 3;
//     std::optional<Activation> activation = Activation::RELU;
//     bool use_bias = true;

//     PerDeviceFFHandle handle;
//     cudnnCreate(&handle.dnn);
//     cublasCreate(&handle.blas);
//     handle.workSpaceSize = 1024 * 1024;
//     std::cout << "Allocating workspace" << std::endl;
//     cudaMalloc(&handle.workSpace, handle.workSpaceSize);
//     std::cout << "Allowing tensor op math conversion" << std::endl;
//     handle.allowTensorOpMathConversion = true;

//     Allocator allocator = get_local_memory_allocator();
//     float *one_ptr;
//     cudaMalloc(&one_ptr, sizeof(float) * batch_size);
//     std::vector<float> host_one(batch_size, 1.0f);
//     cudaMemcpy(one_ptr, host_one.data(), sizeof(float) * batch_size,
//                cudaMemcpyHostToDevice);

//     std::cout << "Init kernel" << std::endl;
    
//     LinearPerDeviceState state = Kernels::Linear::init_kernel(
//         handle, one_ptr, activation, std::nullopt, use_bias, DataType::FLOAT,
//         DataType::FLOAT, DataType::FLOAT, batch_size, in_dim);

//     std::cout << "Init kernel done" << std::endl;
//     float *input_data = static_cast<float *>(
//         allocator.allocate(batch_size * in_dim * sizeof(float)));
//     float *output_data = static_cast<float *>(
//         allocator.allocate(batch_size * out_dim * sizeof(float)));
//     float *weight_data = static_cast<float *>(
//         allocator.allocate(in_dim * out_dim * sizeof(float)));
//     float *bias_data =
//         static_cast<float *>(allocator.allocate(out_dim * sizeof(float)));

//     // Initialize data
//     std::vector<float> host_input_data(batch_size * in_dim, 1.0f);
//     std::vector<float> host_weight_data(in_dim * out_dim, 1.0f);
//     std::vector<float> host_bias_data(out_dim, 1.0f);

//     checkCUDA(cudaMemcpy(input_data, host_input_data.data(),
//                          batch_size * in_dim * sizeof(float),
//                          cudaMemcpyHostToDevice));
//     checkCUDA(cudaMemcpy(weight_data, host_weight_data.data(),
//                         in_dim * out_dim * sizeof(float),
//                         cudaMemcpyHostToDevice));
//     checkCUDA(cudaMemcpy(bias_data, host_bias_data.data(),
//                         out_dim * sizeof(float), cudaMemcpyHostToDevice));
    
//     cudaStream_t stream;
//     checkCUDA(cudaStreamCreate(&stream));

//     std::cout << "Forward pass" << std::endl;
//     // Forward pass
//     Kernels::Linear::forward_kernel(stream, state, input_data, output_data,
//                                     weight_data, use_bias ? bias_data : nullptr,
//                                     in_dim, out_dim, batch_size);
//     std::cout << "Forward pass done" << std::endl;

//     std::vector<float> host_output_data(batch_size * out_dim);
//     checkCUDA(cudaMemcpy(host_output_data.data(), output_data,
//                          batch_size * out_dim * sizeof(float),
//                          cudaMemcpyDeviceToHost));

//     // Backward pass
//     float *input_grad_data = static_cast<float *>(
//         allocator.allocate(batch_size * in_dim * sizeof(float)));
//     float *output_grad_data = static_cast<float *>(
//         allocator.allocate(batch_size * out_dim * sizeof(float)));

//     std::vector<float> host_output_grad_data(batch_size * out_dim, 1.0f);
//     checkCUDA(cudaMemcpy(output_grad_data, host_output_grad_data.data(),
//                          batch_size * out_dim * sizeof(float),
//                          cudaMemcpyHostToDevice));

//     std::cout << "Backward pass" << std::endl;
//     Kernels::Linear::backward_kernel(stream, state, input_data, input_grad_data,
//                                      output_data, output_grad_data, weight_data,
//                                      nullptr, use_bias ? bias_data : nullptr,
//                                      in_dim, out_dim, batch_size);

//     std::cout << "Backward pass done" << std::endl;
//     std::vector<float> host_input_grad_data(batch_size * in_dim);
//     checkCUDA(cudaMemcpy(host_input_grad_data.data(), input_grad_data,
//                          batch_size * in_dim * sizeof(float),
//                          cudaMemcpyDeviceToHost));

//     checkCUDA(cudaStreamDestroy(stream));
//     cudaFree(one_ptr);
//   }
// }
