#include "local-execution/local_task_argument_accessor.h"
#include "kernels/device_handle_t.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/profiling_settings.dtg.h"
#include "op-attrs/ops/input_attrs.dtg.h"
#include "task-spec/task_argument_accessor/task_tensor_parameter.h"
#include "task-spec/task_impl_function.dtg.h"
#include "utils/fmt/variant.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/positive_int/positive_int.h"
#include <doctest/doctest.h>
#include <optional>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("LocalTaskArgumentAccessor") {
    Allocator allocator = create_local_cpu_memory_allocator();
    positive_int embed_dim = 32_p;
    positive_int num_heads = 10_p;

    positive_int batch_size = 40_p;
    positive_int seq_len = 48_p;
    positive_int feature_size = 36_p;

    DataType dtype = DataType::FLOAT;
    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, seq_len, feature_size}},
        DataType::FLOAT,
    };

    GenericTensorAccessorW input =
        allocator.allocate_tensor(input_tensor_shape);
    GenericTensorAccessorW input_grad =
        allocator.allocate_tensor(input_tensor_shape);

    enum Slots {
      INPUT,
      VARIADIC_TENSORS,
    };

    std::unordered_map<TaskTensorParameter, DynamicTensorAccessor>
        tensor_slots_backing = {
            {
                make_task_tensor_parameter_fwd(TensorSlotName::LHS_INPUT),
                DynamicTensorAccessor{input},
            },
            {
                make_task_tensor_parameter_grad(TensorSlotName::LHS_INPUT),
                DynamicTensorAccessor{input_grad},
            },
        };

    global_device_id_t device_idx = global_device_id_t{
        /*coord=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*device_type=*/DeviceType::CPU,
    };

    LocalTaskArgumentAccessor acc = LocalTaskArgumentAccessor{
        /*allocator=*/allocator,
        /*tensor_slots_backing=*/tensor_slots_backing,
        /*profiling_settings=*/ProfilingSettings{0_n, 1_p},
        /*ff_handle=*/cpu_make_device_handle_t(),
        /*op_attrs=*/PCGOperatorAttrs{InputAttrs{input_tensor_shape}},
        /*loss_attrs=*/std::nullopt,
        /*per_device_op_state=*/std::nullopt,
        /*optimizer_attrs=*/std::nullopt,
        /*device_idx=*/device_idx,
    };

    SUBCASE("get_tensor") {
      SUBCASE("get_tensor for read-only forward tensor") {
        GenericTensorAccessor correct = GenericTensorAccessor{
            read_only_accessor_from_write_accessor(input)};
        GenericTensorAccessor result = acc.get_tensor(
            make_task_tensor_parameter_fwd(TensorSlotName::LHS_INPUT),
            Permissions::RO);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor for read-only gradient tensor") {
        GenericTensorAccessor correct = GenericTensorAccessor{
            read_only_accessor_from_write_accessor(input_grad)};
        GenericTensorAccessor result = acc.get_tensor(
            make_task_tensor_parameter_grad(TensorSlotName::LHS_INPUT),
            Permissions::RO);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor for write-only forward tensor") {
        GenericTensorAccessor correct = GenericTensorAccessor{input};
        GenericTensorAccessor result = acc.get_tensor(
            make_task_tensor_parameter_fwd(TensorSlotName::LHS_INPUT),
            Permissions::WO);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor for write-only gradient tensor") {
        GenericTensorAccessor correct = GenericTensorAccessor{input_grad};
        GenericTensorAccessor result = acc.get_tensor(
            make_task_tensor_parameter_grad(TensorSlotName::LHS_INPUT),
            Permissions::WO);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor for read-write forward tensor") {
        GenericTensorAccessor correct = GenericTensorAccessor{input};
        GenericTensorAccessor result = acc.get_tensor(
            make_task_tensor_parameter_fwd(TensorSlotName::LHS_INPUT),
            Permissions::RW);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor for read-write gradient tensor") {
        GenericTensorAccessor correct = GenericTensorAccessor{input_grad};
        GenericTensorAccessor result = acc.get_tensor(
            make_task_tensor_parameter_grad(TensorSlotName::LHS_INPUT),
            Permissions::RW);
        CHECK(correct == result);
      }
    }
  }
}
