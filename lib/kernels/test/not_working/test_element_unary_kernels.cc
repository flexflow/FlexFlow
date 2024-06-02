// #include "doctest/doctest.h"
// #include "kernels/element_unary_kernels.h"
// #include "kernels/local_allocator.h"
// #include <algorithm>
// #include <cmath>
// #include <vector>

// using namespace ::FlexFlow;
// TEST_SUITE(FF_TEST_SUITE) {
//   TEST_CASE("Test Element Unary Forward and Backward Kernel") {
//     std::size_t num_elements = 100;
//     std::size_t dims[] = {10, 10};
//     ArrayShape shape(dims, 2);

//     OperatorType op_type = OperatorType::EXP;
//     ElementScalarUnaryAttrs scalar_attrs = {Op::MUL, 0.5};
//     ElementUnaryUnifiedAttrs attrs = scalar_attrs;

//     PerDeviceFFHandle handle;
//     cudnnCreate(&handle.dnn);
//     cublasCreate(&handle.blas);
//     handle.workSpaceSize = 1024 * 1024;
//     cudaMalloc(&handle.workSpace, handle.workSpaceSize);
//     handle.allowTensorOpMathConversion = true;

//     Allocator allocator = get_local_memory_allocator();
//     ElementUnaryPerDeviceState state =
//         Kernels::ElementUnary::init_kernel(shape, shape, attrs);

//     float *input_data =
//         static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
//     const GenericTensorAccessorR input_accessor{DataType::FLOAT, shape,
//                                                 input_data};

//     float *output_data =
//         static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
//     const GenericTensorAccessorW output_accessor{DataType::FLOAT, shape,
//                                                  output_data};
//     const GenericTensorAccessorR output_read_accessor{DataType::FLOAT, shape,
//                                                       output_data};

//     std::vector<float> host_input_data(num_elements);
//     std::generate(host_input_data.begin(), host_input_data.end(),
//                   []() { return static_cast<float>(rand()) / RAND_MAX; });
//     checkCUDA(cudaMemcpy(input_data, host_input_data.data(),
//                          num_elements * sizeof(float), cudaMemcpyHostToDevice));

//     ffStream_t stream;
//     checkCUDA(cudaStreamCreate(&stream));

//     // Forward pass
//     Kernels::ElementUnary::forward_kernel(stream, state, attrs, handle,
//                                           input_accessor, output_accessor);

//     std::vector<float> host_output_data(num_elements);
//     checkCUDA(cudaMemcpy(host_output_data.data(), output_data,
//                          num_elements * sizeof(float), cudaMemcpyDeviceToHost));

//     // Setup for backward pass
//     float *grad_output_data =
//         static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
//     std::fill_n(host_output_data.begin(), num_elements, 1.0f);
//     checkCUDA(cudaMemcpy(grad_output_data, host_output_data.data(),
//                          num_elements * sizeof(float), cudaMemcpyHostToDevice));
//     GenericTensorAccessorR grad_output_accessor{DataType::FLOAT, shape,
//                                                 grad_output_data};

//     float *grad_input_data =
//         static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
//     std::vector<float> grad_data(num_elements, 0.0f);
//     checkCUDA(cudaMemcpy(grad_input_data, grad_data.data(),
//                          num_elements * sizeof(float), cudaMemcpyHostToDevice));
//     GenericTensorAccessorW grad_input_accessor{DataType::FLOAT, shape,
//                                                 grad_input_data};

//     Kernels::ElementUnary::backward_kernel(stream, state, attrs, handle,
//                                            input_accessor, grad_input_accessor,
//                                            output_read_accessor, grad_output_accessor);

//     std::vector<float> host_grad_input_data(num_elements);
//     checkCUDA(cudaMemcpy(host_grad_input_data.data(), grad_input_data,
//                          num_elements * sizeof(float), cudaMemcpyDeviceToHost));

//     checkCUDA(cudaStreamDestroy(stream));
//   }
// } // namespace FlexFlow
