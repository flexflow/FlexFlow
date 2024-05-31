#include "doctest/doctest.h"
#include "kernels/cast_kernels.h"
#include "kernels/local_allocator.h"
#include <random>
#include <type_traits>

namespace FlexFlow {
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test cast kernel float to double") {
    std::size_t dims[] = {100, 100};
    std::size_t num_dims = 2;
    FlexFlow::ArrayShape shape(dims, num_dims);

    Allocator float_allocator = get_local_memory_allocator();
    void *float_data_ptr =
        float_allocator.allocate((100 * 100) * sizeof(float));
    const GenericTensorAccessorR accessorR{DataType::FLOAT, shape,
                                           float_data_ptr};

    Allocator double_allocator = get_local_memory_allocator();
    void *double_data_ptr =
        double_allocator.allocate((100 * 100) * sizeof(double));
    const GenericTensorAccessorW accessorW{DataType::DOUBLE, shape,
                                           double_data_ptr};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> host_data(100 * 100);

    for (auto &val : host_data) {
      val = dist(gen);
    }

    checkCUDA(cudaMemcpy(float_data_ptr, host_data.data(),
                         host_data.size() * sizeof(float),
                         cudaMemcpyHostToDevice));

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    Kernels::Cast::forward_kernel(nullptr, accessorR, accessorW,
                                  DataType::FLOAT, DataType::DOUBLE);

    std::vector<float> host_float_data(100 * 100);
    std::vector<double> host_double_data(100 * 100);

    checkCUDA(cudaMemcpy(host_float_data.data(), float_data_ptr,
                         host_float_data.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(host_double_data.data(), double_data_ptr,
                         host_double_data.size() * sizeof(double),
                         cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < host_float_data.size(); ++i) {
      REQUIRE(typeid(host_double_data[i]) == typeid(double));
    }

    checkCUDA(cudaStreamDestroy(stream));
  }

  TEST_CASE("Test cast kernel Int to Float") {
    std::size_t dims[] = {100, 100};
    std::size_t num_dims = 2;
    FlexFlow::ArrayShape shape(dims, num_dims);

    Allocator int_allocator = get_local_memory_allocator();
    void *int_data_ptr = int_allocator.allocate((100 * 100) * sizeof(int));
    const GenericTensorAccessorR accessorR{DataType::INT32, shape,
                                           int_data_ptr};

    Allocator float_allocator = get_local_memory_allocator();
    void *float_data_ptr =
        float_allocator.allocate((100 * 100) * sizeof(float));
    const GenericTensorAccessorW accessorW{DataType::FLOAT, shape,
                                           float_data_ptr};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 1);

    std::vector<int> host_data(100 * 100);
    for (auto &val : host_data) {
      val = dist(gen);
    }

    checkCUDA(cudaMemcpy(int_data_ptr, host_data.data(),
                         host_data.size() * sizeof(int),
                         cudaMemcpyHostToDevice));

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    Kernels::Cast::forward_kernel(nullptr, accessorR, accessorW,
                                  DataType::INT32, DataType::FLOAT);

    std::vector<int> host_int_data(100 * 100);
    std::vector<float> host_float_data(100 * 100);

    checkCUDA(cudaMemcpy(host_int_data.data(), int_data_ptr,
                         host_int_data.size() * sizeof(int),
                         cudaMemcpyDeviceToHost));

    checkCUDA(cudaMemcpy(host_float_data.data(), float_data_ptr,
                         host_float_data.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < host_int_data.size(); ++i) {
      REQUIRE(typeid(host_float_data[i]) == typeid(float));
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
} // namespace FlexFlow