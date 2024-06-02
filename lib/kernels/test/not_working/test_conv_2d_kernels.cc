// #include "doctest/doctest.h"
// #include "kernels/conv_2d_kernels.h"
// #include "kernels/local_allocator.h"
// #include <algorithm>
// #include <iostream>
// #include <vector>

// using namespace ::FlexFlow;

// TEST_SUITE(FF_TEST_SUITE) {
//   TEST_CASE("Test Conv2D Forward and Backward Kernel") {
//     std::size_t batch_size = 1;
//     std::size_t num_channels = 1;
//     std::size_t height = 10;
//     std::size_t width = 10;
//     std::size_t num_filters = 64;
//     std::size_t output_height = 8;  // Calculated or expected based on the padding and stride
//     std::size_t output_width = 8;
//     std::size_t kernel_h = 3, kernel_w = 3;
//     int pad_h = 1, pad_w = 1;
//     int stride_h = 1, stride_w = 1;
//     int groups = 1;

//     std::size_t num_input_elements = batch_size * num_channels * height * width;
//     std::size_t num_output_elements = batch_size * num_filters * output_height * output_width;

//     ArrayShape input_shape({batch_size, num_channels, height, width});
//     ArrayShape output_shape({batch_size, num_filters, output_height, output_width});
//     ArrayShape filter_shape({num_filters, num_channels, kernel_h, kernel_w});

//     PerDeviceFFHandle handle;
//     cudnnCreate(&handle.dnn);
//     cublasCreate(&handle.blas);
//     handle.workSpaceSize = 1024 * 1024 * 64;
//     cudaMalloc(&handle.workSpace, handle.workSpaceSize);
//     handle.allowTensorOpMathConversion = true;

//     Allocator allocator = get_local_memory_allocator();
//     float *filter_ptr =
//         static_cast<float *>(allocator.allocate(num_filters * num_channels * kernel_h * kernel_w * sizeof(float)));
//     float *filter_grad_ptr =
//         static_cast<float *>(allocator.allocate(num_filters * num_channels * kernel_h * kernel_w * sizeof(float)));
//     float *input_data =
//         static_cast<float *>(allocator.allocate(num_input_elements * sizeof(float)));
//     float *output_data =
//         static_cast<float *>(allocator.allocate(num_output_elements * sizeof(float)));

//     std::vector<float> host_input_data(num_input_elements);
//     std::generate(host_input_data.begin(), host_input_data.end(),
//                   []() { return static_cast<float>(rand()) / RAND_MAX; });
//     checkCUDA(cudaMemcpy(input_data, host_input_data.data(),
//                          num_input_elements * sizeof(float), cudaMemcpyHostToDevice));
//     const GenericTensorAccessorR input_accessor{DataType::FLOAT, input_shape, input_data};
//     const GenericTensorAccessorW output_accessor{DataType::FLOAT, output_shape, output_data};

//     Conv2DPerDeviceState state = Kernels::Conv2D::init_kernel(
//         handle, {}, kernel_h, kernel_w, groups, pad_h, pad_w, stride_h,
//         stride_w, input_accessor, output_accessor, filter_ptr, filter_grad_ptr);

//     cudaStream_t stream;
//     checkCUDA(cudaStreamCreate(&stream));

//     // Forward pass
//     Kernels::Conv2D::forward_kernel(stream, state, input_data, output_data, filter_ptr, nullptr, {});

//     std::vector<float> host_output_data(num_output_elements);
//     checkCUDA(cudaMemcpy(host_output_data.data(), output_data,
//                          num_output_elements * sizeof(float), cudaMemcpyDeviceToHost));

//     // Verify output - ensure some computation happened
//     for (auto &val : host_output_data) {
//       CHECK(val != 0);
//     }

//     // Backward pass
//     float *input_grad_data =
//         static_cast<float *>(allocator.allocate(num_input_elements * sizeof(float)));
//     float *output_grad_data =
//         static_cast<float *>(allocator.allocate(num_output_elements * sizeof(float)));

//     // Initialize gradients to propagate back
//     std::fill_n(host_output_data.begin(), num_output_elements, 1.0f);
//     checkCUDA(cudaMemcpy(output_grad_data, host_output_data.data(),
//                          num_output_elements * sizeof(float), cudaMemcpyHostToDevice));

//     Kernels::Conv2D::backward_kernel(stream, state, input_data, input_grad_data,
//                                      output_data, output_grad_data, filter_ptr,
//                                      filter_grad_ptr, nullptr, {});

//     std::vector<float> host_input_grad_data(num_input_elements);
//     checkCUDA(cudaMemcpy(host_input_grad_data.data(), input_grad_data,
//                          num_input_elements * sizeof(float), cudaMemcpyDeviceToHost));

//     // Verify input gradients
//     for (auto &val : host_input_grad_data) {
//       CHECK(val != 0);
//     }

//     checkCUDA(cudaStreamDestroy(stream));
//   }
// }
