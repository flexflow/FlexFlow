#include "doctest/doctest.h"
#include "kernels/layer_norm_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test LayerNorm Forward and Backward Kernel") {
    size_t batch_size = 10;
    size_t feature_size = 10;
    size_t num_elements = batch_size * feature_size;
    float epsilon = 1e-5f;
    bool elementwise_affine = true;

    ArrayShape shape = ArrayShape{
        std::vector<size_t>{batch_size, feature_size},
    };

    ArrayShape feature_shape = ArrayShape{
        std::vector<size_t>{feature_size},
    };

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    LayerNormPerDeviceState state =
        Kernels::LayerNorm::init_kernel(handle,
                                        allocator,
                                        elementwise_affine,
                                        batch_size,
                                        feature_size,
                                        epsilon);

    float *input_data, *output_data, *gamma_data, *beta_data;
    std::vector<float **> ptrs = {
        &input_data, &output_data, &gamma_data, &beta_data};
    std::vector<size_t> sizes = {
        num_elements, num_elements, feature_size, feature_size};

    allocate_ptrs(ptrs, sizes, allocator);

    fillDeviceDataNum(&input_data, num_elements, 1.0f);
    fillDeviceDataNum(&gamma_data, feature_size, 1.0f);
    fillDeviceDataNum(&beta_data, feature_size, 0.0f);
    randomFillDeviceData(&input_data, num_elements);

    GenericTensorAccessorR input_accessor{DataType::FLOAT, shape, input_data};
    GenericTensorAccessorW output_accessor{DataType::FLOAT, shape, output_data};
    GenericTensorAccessorW gamma_accessor{
        DataType::FLOAT, feature_shape, gamma_data};
    GenericTensorAccessorR gamma_accessor_read{
        DataType::FLOAT, feature_shape, gamma_data};
    GenericTensorAccessorW beta_accessor{
        DataType::FLOAT, feature_shape, beta_data};

    SUBCASE("Test Layer Norm Forward") {
      Kernels::LayerNorm::forward_kernel(stream,
                                         state,
                                         input_accessor,
                                         output_accessor,
                                         gamma_accessor,
                                         beta_accessor);

      std::vector<float> host_output_data(num_elements);
      checkCUDA(cudaMemcpy(host_output_data.data(),
                           output_data,
                           num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));
    }

    SUBCASE("Test Layer Norm Backward") {
      float *grad_output_data, *grad_input_data, *gamma_grad_data,
          *beta_grad_data;
      std::vector<float **> ptrs_grad = {&grad_output_data,
                                         &grad_input_data,
                                         &gamma_grad_data,
                                         &beta_grad_data};
      std::vector<size_t> sizes_grad = {
          num_elements, num_elements, feature_size, feature_size};

      allocate_ptrs(ptrs_grad, sizes_grad, allocator);
      fillDeviceDataNum(&grad_output_data, num_elements, 1.0f);

      GenericTensorAccessorR grad_output_accessor{
          DataType::FLOAT, shape, grad_output_data};
      GenericTensorAccessorW grad_input_accessor{
          DataType::FLOAT, shape, grad_input_data};
      GenericTensorAccessorW gamma_grad_accessor{
          DataType::FLOAT, feature_shape, gamma_grad_data};
      GenericTensorAccessorW beta_grad_accessor{
          DataType::FLOAT, feature_shape, beta_grad_data};

      Kernels::LayerNorm::backward_kernel(stream,
                                          state,
                                          grad_output_accessor,
                                          input_accessor,
                                          grad_input_accessor,
                                          gamma_accessor_read,
                                          gamma_grad_accessor,
                                          beta_grad_accessor);
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
