#include "doctest/doctest.h"
#include "local-execution/local_cpu_allocator.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/ops/attention.h"
#include "local-execution/task_signature_impl.h"

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Local Task Argument Accessor") {
    Allocator allocator = create_local_cpu_memory_allocator();
    int embed_dim = 32;
    int num_heads = 10;

    size_t batch_size = 40;
    size_t seq_len = 48;
    size_t feature_size = 36;

    DataType dtype = DataType::FLOAT;
    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered<size_t>{batch_size, seq_len, feature_size}},
        DataType::FLOAT,
    };

    GenericTensorAccessorW query =
        allocator.allocate_tensor(input_tensor_shape);
    GenericTensorAccessorW query_grad =
        allocator.allocate_tensor(input_tensor_shape);

    std::vector<GenericTensorAccessorW> mock_variadic_tensors = {query, query};
    std::vector<GenericTensorAccessorW> mock_variadic_tensors_grad = {
        query_grad, query_grad};

    enum MockSlots {
      QUERY,
      MOCK_VARIADIC_TENSORS,
    };

    TensorSlotsBacking mock_tensor_slots_backing = {
        {{QUERY, IsGrad::NO}, query},
        {{QUERY, IsGrad::YES}, query_grad},
        {{MOCK_VARIADIC_TENSORS, IsGrad::NO}, mock_variadic_tensors},
        {{MOCK_VARIADIC_TENSORS, IsGrad::YES}, mock_variadic_tensors_grad},
    };

    LocalTaskArgumentAccessor acc = {allocator, mock_tensor_slots_backing, {}};

    SUBCASE("get_tensor") {
      SUBCASE("Read-only query tensor") {
        GenericTensorAccessorR correct =
            read_only_accessor_from_write_accessor(query);
        GenericTensorAccessorR result = std::get<GenericTensorAccessorR>(
            acc.get_tensor(QUERY, Permissions::RO, IsGrad::NO));
        CHECK(correct == result);
      }
      SUBCASE("Read-only query grad tensor") {
        GenericTensorAccessorR correct =
            read_only_accessor_from_write_accessor(query_grad);
        GenericTensorAccessorR result = std::get<GenericTensorAccessorR>(
            acc.get_tensor(QUERY, Permissions::RO, IsGrad::YES));
        CHECK(correct == result);
      }
      SUBCASE("Write-only query tensor") {
        GenericTensorAccessorW result = std::get<GenericTensorAccessorW>(
            acc.get_tensor(QUERY, Permissions::WO, IsGrad::NO));
        CHECK(query == result);
      }
      SUBCASE("Write-only query grad tensor") {
        GenericTensorAccessorW result = std::get<GenericTensorAccessorW>(
            acc.get_tensor(QUERY, Permissions::WO, IsGrad::YES));
        CHECK(query_grad == result);
      }
      SUBCASE("Read-write query tensor") {
        GenericTensorAccessorW result = std::get<GenericTensorAccessorW>(
            acc.get_tensor(QUERY, Permissions::RW, IsGrad::NO));
        CHECK(query == result);
      }
      SUBCASE("Read-write query grad tensor") {
        GenericTensorAccessorW result = std::get<GenericTensorAccessorW>(
            acc.get_tensor(QUERY, Permissions::RW, IsGrad::YES));
        CHECK(query_grad == result);
      }
    }

    SUBCASE("get_tensor") {
      SUBCASE("Read-only mock tensors") {
        std::vector<GenericTensorAccessorR> correct = {
            read_only_accessor_from_write_accessor(mock_variadic_tensors.at(0)),
            read_only_accessor_from_write_accessor(
                mock_variadic_tensors.at(1))};
        std::vector<GenericTensorAccessorR> result =
            std::get<std::vector<GenericTensorAccessorR>>(
                acc.get_variadic_tensor(
                    MOCK_VARIADIC_TENSORS, Permissions::RO, IsGrad::NO));
        CHECK(correct == result);
      }
      SUBCASE("Read-only mock grad tensors") {
        std::vector<GenericTensorAccessorR> correct = {
            read_only_accessor_from_write_accessor(
                mock_variadic_tensors_grad.at(0)),
            read_only_accessor_from_write_accessor(
                mock_variadic_tensors_grad.at(1))};
        std::vector<GenericTensorAccessorR> result =
            std::get<std::vector<GenericTensorAccessorR>>(
                acc.get_variadic_tensor(
                    MOCK_VARIADIC_TENSORS, Permissions::RO, IsGrad::YES));
        CHECK(correct == result);
      }
      SUBCASE("Write-only mock tensors") {
        std::vector<GenericTensorAccessorW> result =
            std::get<std::vector<GenericTensorAccessorW>>(
                acc.get_variadic_tensor(
                    MOCK_VARIADIC_TENSORS, Permissions::WO, IsGrad::NO));
        CHECK(mock_variadic_tensors == result);
      }
      SUBCASE("Write-only mock grad tensors") {
        std::vector<GenericTensorAccessorW> result =
            std::get<std::vector<GenericTensorAccessorW>>(
                acc.get_variadic_tensor(
                    MOCK_VARIADIC_TENSORS, Permissions::WO, IsGrad::YES));
        CHECK(mock_variadic_tensors_grad == result);
      }
      SUBCASE("Read-write mock tensors") {
        std::vector<GenericTensorAccessorW> result =
            std::get<std::vector<GenericTensorAccessorW>>(
                acc.get_variadic_tensor(
                    MOCK_VARIADIC_TENSORS, Permissions::RW, IsGrad::NO));
        CHECK(mock_variadic_tensors == result);
      }
      SUBCASE("Read-write mock grad tensors") {
        std::vector<GenericTensorAccessorW> result =
            std::get<std::vector<GenericTensorAccessorW>>(
                acc.get_variadic_tensor(
                    MOCK_VARIADIC_TENSORS, Permissions::RW, IsGrad::YES));
        CHECK(mock_variadic_tensors_grad == result);
      }
    }
  }
}

} // namespace FlexFlow
