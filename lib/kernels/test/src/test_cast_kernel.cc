#include "doctest/doctest.h"
#include "kernels/cast_kernels.h"
#include "test_utils.h"
#include <type_traits>

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test cast kernel float to double") {
    ArrayShape shape = ArrayShape{
        std::vector<size_t>{100, 100},
    };

    Allocator allocator = get_local_memory_allocator();

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    void *float_data_ptr, *double_data_ptr;
    std::vector<void **> ptrs = {&float_data_ptr, &double_data_ptr};
    std::vector<size_t> sizes = {(100 * 100), (100 * 100)};
    allocate_ptrs(ptrs, sizes, allocator);
    randomFillDeviceData(&float_data_ptr, 100 * 100);

    GenericTensorAccessorR accessorR{DataType::FLOAT, shape, float_data_ptr};
    GenericTensorAccessorW accessorW{DataType::DOUBLE, shape, double_data_ptr};

    Kernels::Cast::forward_kernel(
        nullptr, accessorR, accessorW, DataType::FLOAT, DataType::DOUBLE);

    std::vector<float> host_float_data(100 * 100);
    std::vector<double> host_double_data(100 * 100);

    checkCUDA(cudaMemcpy(host_float_data.data(),
                         float_data_ptr,
                         host_float_data.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(host_double_data.data(),
                         double_data_ptr,
                         host_double_data.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < host_float_data.size(); ++i) {
      REQUIRE(typeid(host_double_data[i]) == typeid(double));
    }

    checkCUDA(cudaStreamDestroy(stream));
  }

  TEST_CASE("Test cast kernel Int to Float") {
    ArrayShape shape = ArrayShape{
        std::vector<size_t>{100, 100},
    };

    Allocator allocator = get_local_memory_allocator();

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    void *int_data_ptr, *float_data_ptr;
    std::vector<void **> ptrs = {&int_data_ptr, &float_data_ptr};
    std::vector<size_t> sizes = {(100 * 100), (100 * 100)};
    allocate_ptrs(ptrs, sizes, allocator);
    randomFillDeviceData(&int_data_ptr, 100 * 100);

    GenericTensorAccessorR accessorR{DataType::INT32, shape, int_data_ptr};
    GenericTensorAccessorW accessorW{DataType::FLOAT, shape, float_data_ptr};

    Kernels::Cast::forward_kernel(
        nullptr, accessorR, accessorW, DataType::INT32, DataType::FLOAT);

    std::vector<int> host_int_data(100 * 100);
    std::vector<float> host_float_data(100 * 100);

    checkCUDA(cudaMemcpy(host_int_data.data(),
                         int_data_ptr,
                         host_int_data.size() * sizeof(int),
                         cudaMemcpyDeviceToHost));

    checkCUDA(cudaMemcpy(host_float_data.data(),
                         float_data_ptr,
                         host_float_data.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    checkCUDA(cudaStreamDestroy(stream));
  }
}
