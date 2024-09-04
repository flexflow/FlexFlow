#include "doctest/doctest.h"
#include "local-execution/local_cpu_allocator.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/task_signature_impl.h"
#include "utils/fmt/variant.h"

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("LocalTaskArgumentAccessor") {
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
      SUBCASE("get_tensor(slot_id_t, Permissions::RO, IsGrad::NO)") {
        GenericTensorAccessor correct = GenericTensorAccessor{
            read_only_accessor_from_write_accessor(input)};
        GenericTensorAccessor result =
            acc.get_tensor(slot_id_t{INPUT}, Permissions::RO, IsGrad::NO);
        CHECK(correct == result);
      }
      SUBCASE("get_tensor(slot_id_t, Permissions::RO, IsGrad::YES)") {
        GenericTensorAccessor correct = GenericTensorAccessor{
            read_only_accessor_from_write_accessor(input_grad)};
        GenericTensorAccessor result =
            acc.get_tensor(slot_id_t{INPUT}, Permissions::RO, IsGrad::YES);
        CHECK(correct == result);
      }
      SUBCASE("get_tensor(slot_id_t, Permissions::WO, IsGrad::NO)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input};
        GenericTensorAccessor result =
            acc.get_tensor(slot_id_t{INPUT}, Permissions::WO, IsGrad::NO);
        CHECK(correct == result);
      }
      SUBCASE("get_tensor(slot_id_t, Permissions::WO, IsGrad::YES)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input_grad};
        GenericTensorAccessor result =
            acc.get_tensor(slot_id_t{INPUT}, Permissions::WO, IsGrad::YES);
        CHECK(correct == result);
      }
      SUBCASE("get_tensor(slot_id_t, Permissions::RW, IsGrad::NO)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input};
        GenericTensorAccessor result =
            acc.get_tensor(slot_id_t{INPUT}, Permissions::RW, IsGrad::NO);
        CHECK(correct == result);
      }
      SUBCASE("get_tensor(slot_id_t, Permissions::RW, IsGrad::YES)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input_grad};
        GenericTensorAccessor result =
            acc.get_tensor(slot_id_t{INPUT}, Permissions::RW, IsGrad::YES);
        CHECK(correct == result);
      }
    }

    SUBCASE("get_variadic_tensor") {
      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::RO, IsGrad::NO)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{std::vector<GenericTensorAccessorR>{
                read_only_accessor_from_write_accessor(variadic_tensors.at(0)),
                read_only_accessor_from_write_accessor(
                    variadic_tensors.at(1))}};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::RO, IsGrad::NO);
        CHECK(result == correct);
      }
      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::RO, IsGrad::YES)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{std::vector<GenericTensorAccessorR>{
                read_only_accessor_from_write_accessor(
                    variadic_tensors_grad.at(0)),
                read_only_accessor_from_write_accessor(
                    variadic_tensors_grad.at(1))}};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::RO, IsGrad::YES);
        CHECK(result == correct);
      }
      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::WO, IsGrad::NO)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::WO, IsGrad::NO);
        CHECK(result == correct);
      }
      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::WO, IsGrad::YES)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors_grad};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::WO, IsGrad::YES);
        CHECK(result == correct);
      }
      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::WO, IsGrad::NO)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::RW, IsGrad::NO);
        CHECK(result == correct);
      }
      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::WO, IsGrad::YES)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors_grad};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::RW, IsGrad::YES);
        CHECK(result == correct);
      }
    }
  }
}

} // namespace FlexFlow
