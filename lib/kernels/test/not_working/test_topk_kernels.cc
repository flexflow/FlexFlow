// #include "doctest/doctest.h"
// #include "kernels/local_allocator.h"
// #include "kernels/topk_kernels.h"
// #include <algorithm>
// #include <random>
// #include <vector>

// using namespace ::FlexFlow;
// TEST_SUITE(FF_TEST_SUITE) {
//   TEST_CASE("Test TopK Forward and Backward Kernel") {
//     std::size_t num_elements = 100; // Total elements in a single batch
//     std::size_t batch_size = 10;    // Number of batches
//     int k = 5;                      // Top 'k' elements to find
//     bool sorted = true;             // Whether the output should be

//     PerDeviceFFHandle handle;
//     cudnnCreate(&handle.dnn);
//     cublasCreate(&handle.blas);
//     handle.workSpaceSize = 1024 * 1024;
//     cudaMalloc(&handle.workSpace, handle.workSpaceSize);
//     handle.allowTensorOpMathConversion = true;

//     Allocator allocator = get_local_memory_allocator();
//     TopKPerDeviceState state = Kernels::TopK::init_kernel(sorted);

//     float *input_data = static_cast<float *>(
//         allocator.allocate(batch_size * num_elements * sizeof(float)));
//     float *output_data = static_cast<float *>(
//         allocator.allocate(batch_size * k * sizeof(float)));
//     int *indices_data =
//         static_cast<int *>(allocator.allocate(batch_size * k * sizeof(int)));

//     // Generate random input data
//     std::mt19937 gen(12345);
//     std::uniform_real_distribution<> dis(0.0, 1.0);
//     std::vector<float> host_input_data(batch_size * num_elements);
//     std::generate(host_input_data.begin(), host_input_data.end(),
//                   [&]() { return dis(gen); });
//     checkCUDA(cudaMemcpy(input_data, host_input_data.data(),
//                          batch_size * num_elements * sizeof(float),
//                          cudaMemcpyHostToDevice));

//     cudaStream_t stream;
//     checkCUDA(cudaStreamCreate(&stream));

//     // Forward pass
//     Kernels::TopK::forward_kernel(stream, state, input_data, output_data,
//                                   indices_data, batch_size, num_elements, k,
//                                   sorted);

//     std::vector<float> host_output_data(batch_size * k);
//     std::vector<int> host_indices_data(batch_size * k);
//     checkCUDA(cudaMemcpy(host_output_data.data(), output_data,
//                          batch_size * k * sizeof(float),
//                          cudaMemcpyDeviceToHost));
//     checkCUDA(cudaMemcpy(host_indices_data.data(), indices_data,
//                          batch_size * k * sizeof(int), cudaMemcpyDeviceToHost));

//     // Verify output of forward pass
//     for (size_t b = 0; b < batch_size; ++b) {
//       std::vector<float> slice(host_input_data.begin() + b * num_elements,
//                                host_input_data.begin() + (b + 1) * num_elements);
//       std::vector<float> expected_topk(k);
//       std::partial_sort_copy(slice.begin(), slice.end(), expected_topk.begin(),
//                              expected_topk.end(), std::greater<float>());

//       for (int i = 0; i < k; ++i) {
//         CHECK(doctest::Approx(host_output_data[b * k + i]) == expected_topk[i]);
//       }
//     }

//     // Setup for backward pass
//     float *grad_output_data = static_cast<float *>(
//         allocator.allocate(batch_size * k * sizeof(float)));
//     std::fill_n(grad_output_data, batch_size * k, 1.0f);
//     // Assuming gradient from next layer as 1 for simplicity

//     float *in_grad_data = static_cast<float *>(
//         allocator.allocate(batch_size * num_elements * sizeof(float)));
//     std::fill_n(in_grad_data, batch_size * num_elements, 0.0f);

//     // Backward pass
//     Kernels::TopK::backward_kernel(stream, state, grad_output_data,
//                                    indices_data, in_grad_data, batch_size,
//                                    num_elements, k);

//     std::vector<float> host_in_grad_data(batch_size * num_elements);
//     checkCUDA(cudaMemcpy(host_in_grad_data.data(), in_grad_data,
//                          batch_size * num_elements * sizeof(float),
//                          cudaMemcpyDeviceToHost));

//     // Verify output of backward pass
//     for (size_t b = 0; b < batch_size; ++b) {
//       for (int i = 0; i < k; ++i) {
//         int idx = host_indices_data[b * k + i];
//         CHECK(doctest::Approx(host_in_grad_data[b * num_elements + idx]) ==
//               1.0f);
//       }
//     }

//     checkCUDA(cudaStreamDestroy(stream));
//   }
// }
