#include "doctest/doctest.h"
#include "local-execution/local_cpu_allocator.h"
#include "local-execution/local_task_argument_accessor.h"
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

    GenericTensorAccessorW input =
        allocator.allocate_tensor(input_tensor_shape);
    GenericTensorAccessorW input_grad =
        allocator.allocate_tensor(input_tensor_shape);

    std::vector<GenericTensorAccessorW> variadic_tensors = {input, input};
    std::vector<GenericTensorAccessorW> variadic_tensors_grad = {input_grad,
                                                                 input_grad};

    enum Slots {
      INPUT,
      VARIADIC_TENSORS,
    };

    TensorSlotsBacking tensor_slots_backing = {
        {SlotGradId{slot_id_t{INPUT}, IsGrad::NO}, input},
        {SlotGradId{slot_id_t{INPUT}, IsGrad::YES}, input_grad},
        {SlotGradId{slot_id_t{VARIADIC_TENSORS}, IsGrad::NO}, variadic_tensors},
        {SlotGradId{slot_id_t{VARIADIC_TENSORS}, IsGrad::YES},
         variadic_tensors_grad},
    };

    LocalTaskArgumentAccessor acc = {allocator, tensor_slots_backing, {}};

    SUBCASE("get_tensor") {
      SUBCASE("Read-only input tensor") {
        GenericTensorAccessorR correct =
            read_only_accessor_from_write_accessor(input);
        GenericTensorAccessorR result = std::get<GenericTensorAccessorR>(
            acc.get_tensor(slot_id_t{INPUT}, Permissions::RO, IsGrad::NO));
        CHECK(correct == result);
      }
      SUBCASE("Read-only input grad tensor") {
        GenericTensorAccessorR correct =
            read_only_accessor_from_write_accessor(input_grad);
        GenericTensorAccessorR result = std::get<GenericTensorAccessorR>(
            acc.get_tensor(slot_id_t{INPUT}, Permissions::RO, IsGrad::YES));
        CHECK(correct == result);
      }
      SUBCASE("Write-only input tensor") {
        GenericTensorAccessorW result = std::get<GenericTensorAccessorW>(
            acc.get_tensor(slot_id_t{INPUT}, Permissions::WO, IsGrad::NO));
        CHECK(input == result);
      }
      SUBCASE("Write-only input grad tensor") {
        GenericTensorAccessorW result = std::get<GenericTensorAccessorW>(
            acc.get_tensor(slot_id_t{INPUT}, Permissions::WO, IsGrad::YES));
        CHECK(input_grad == result);
      }
      SUBCASE("Read-write input tensor") {
        GenericTensorAccessorW result = std::get<GenericTensorAccessorW>(
            acc.get_tensor(slot_id_t{INPUT}, Permissions::RW, IsGrad::NO));
        CHECK(input == result);
      }
      SUBCASE("Read-write input grad tensor") {
        GenericTensorAccessorW result = std::get<GenericTensorAccessorW>(
            acc.get_tensor(slot_id_t{INPUT}, Permissions::RW, IsGrad::YES));
        CHECK(input_grad == result);
      }
    }

    SUBCASE("get_variadic_tensor") {
      SUBCASE("Read-only tensors") {
        std::vector<GenericTensorAccessorR> correct = {
            read_only_accessor_from_write_accessor(variadic_tensors.at(0)),
            read_only_accessor_from_write_accessor(variadic_tensors.at(1))};
        std::vector<GenericTensorAccessorR> result =
            std::get<std::vector<GenericTensorAccessorR>>(
                acc.get_variadic_tensor(
                    slot_id_t{VARIADIC_TENSORS}, Permissions::RO, IsGrad::NO));
        CHECK(result == correct);
      }
      SUBCASE("Read-only grad tensors") {
        std::vector<GenericTensorAccessorR> correct = {
            read_only_accessor_from_write_accessor(variadic_tensors_grad.at(0)),
            read_only_accessor_from_write_accessor(
                variadic_tensors_grad.at(1))};
        std::vector<GenericTensorAccessorR> result =
            std::get<std::vector<GenericTensorAccessorR>>(
                acc.get_variadic_tensor(
                    slot_id_t{VARIADIC_TENSORS}, Permissions::RO, IsGrad::YES));
        CHECK(correct == result);
      }
      SUBCASE("Write-only tensors") {
        std::vector<GenericTensorAccessorW> result =
            std::get<std::vector<GenericTensorAccessorW>>(
                acc.get_variadic_tensor(
                    slot_id_t{VARIADIC_TENSORS}, Permissions::WO, IsGrad::NO));
        CHECK(variadic_tensors == result);
      }
      SUBCASE("Write-only grad tensors") {
        std::vector<GenericTensorAccessorW> result =
            std::get<std::vector<GenericTensorAccessorW>>(
                acc.get_variadic_tensor(
                    slot_id_t{VARIADIC_TENSORS}, Permissions::WO, IsGrad::YES));
        CHECK(variadic_tensors_grad == result);
      }
      SUBCASE("Read-write tensors") {
        std::vector<GenericTensorAccessorW> result =
            std::get<std::vector<GenericTensorAccessorW>>(
                acc.get_variadic_tensor(
                    slot_id_t{VARIADIC_TENSORS}, Permissions::RW, IsGrad::NO));
        CHECK(variadic_tensors == result);
      }
      SUBCASE("Read-write grad tensors") {
        std::vector<GenericTensorAccessorW> result =
            std::get<std::vector<GenericTensorAccessorW>>(
                acc.get_variadic_tensor(
                    slot_id_t{VARIADIC_TENSORS}, Permissions::RW, IsGrad::YES));
        CHECK(variadic_tensors_grad == result);
      }
    }
  }
}

} // namespace FlexFlow
