#include "doctest/doctest.h"
#include "kernels/layer_norm_kernels.h"
#include "kernels/local_allocator.h"
#include "test_utils.h"
#include <algorithm>
#include <vector>

template <typename T>
void allocate_ptrs(std::vector<T **> &gpu_data_ptrs,
                   const std::vector<size_t> &num_elements,
                   Allocator &allocator) {
  for (size_t i = 0; i < gpu_data_ptrs.size(); ++i) {
    *gpu_data_ptrs[i] =
        static_cast<T *>(allocator.allocate(num_elements[i] * sizeof(float)));
  }
}

using namespace ::FlexFlow;

TEST_SUITE("kernel-tests") {
  TEST_CASE("Test LayerNorm Forward and Backward Kernel") {
    size_t batch_size = 10;
    size_t feature_size = 10;
    size_t dims[] = {batch_size, feature_size};
    size_t feature_dims[] = {feature_size};
    size_t num_elements = batch_size * feature_size;
    float epsilon = 1e-5f;
    bool elementwise_affine = true;

    ArrayShape shape(dims, 2);
    ArrayShape feature_shape(feature_dims, 1);

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    float *input_data, *output_data, *gamma_data, *beta_data;
    std::vector<float **> ptrs = {&input_data, &output_data, &gamma_data,
                                  &beta_data};
    std::vector<size_t> sizes = {num_elements, num_elements, feature_size,
                                 feature_size};

    allocate_ptrs(ptrs, sizes, allocator);

    fillDeviceDataNum(&input_data, num_elements, 1.0f);
    fillDeviceDataNum(&gamma_data, feature_size, 1.0f);
    fillDeviceDataNum(&beta_data, feature_size, 0.0f);
    randomFillDeviceData(&input_data, num_elements);

    const GenericTensorAccessorR input_accessor{DataType::FLOAT, shape,
                                                input_data};
    const GenericTensorAccessorW output_accessor{DataType::FLOAT, shape,
                                                 output_data};
    const GenericTensorAccessorW gamma_accessor{DataType::FLOAT, feature_shape,
                                                gamma_data};
    const GenericTensorAccessorR gamma_accessor_read{DataType::FLOAT,
                                                     feature_shape, gamma_data};
    const GenericTensorAccessorW beta_accessor{DataType::FLOAT, feature_shape,
                                               beta_data};

    LayerNormPerDeviceState state =
        Kernels::LayerNorm::init_kernel(handle, allocator, elementwise_affine,
                                        batch_size, feature_size, epsilon);

    Kernels::LayerNorm::forward_kernel(stream, state, input_accessor,
                                       output_accessor, gamma_accessor,
                                       beta_accessor);

    std::vector<float> host_output_data(num_elements);
    checkCUDA(cudaMemcpy(host_output_data.data(), output_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    float *grad_output_data, *grad_input_data, *gamma_grad_data,
        *beta_grad_data;
    std::vector<float **> ptrs_grad = {&grad_output_data, &grad_input_data,
                                       &gamma_grad_data, &beta_grad_data};
    std::vector<size_t> sizes_grad = {num_elements, num_elements, feature_size,
                                      feature_size};

    allocate_ptrs(ptrs_grad, sizes_grad, allocator);
    fillDeviceDataNum(&grad_output_data, num_elements, 1.0f);

    const GenericTensorAccessorR grad_output_accessor{DataType::FLOAT, shape,
                                                      grad_output_data};
    const GenericTensorAccessorW grad_input_accessor{DataType::FLOAT, shape,
                                                     grad_input_data};
    const GenericTensorAccessorW gamma_grad_accessor{
        DataType::FLOAT, feature_shape, gamma_grad_data};
    const GenericTensorAccessorW beta_grad_accessor{
        DataType::FLOAT, feature_shape, beta_grad_data};

    Kernels::LayerNorm::backward_kernel(
        stream, state, grad_output_accessor, input_accessor,
        grad_input_accessor, gamma_accessor_read, gamma_grad_accessor,
        beta_grad_accessor);
    checkCUDA(cudaStreamDestroy(stream));
  }
}
