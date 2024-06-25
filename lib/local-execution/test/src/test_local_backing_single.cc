#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/test/src/test_utils.h"
#include "local-execution/local_cost_estimator.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Local Backing Single Operator -- Cast") {
    // allocate input memory
    Allocator allocator = get_local_memory_allocator();
    DataType input_dtype = DataType::FLOAT;
    DataType output_dtype = DataType::DOUBLE;
    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered<size_t>{12, 16}},
        input_dtype,
    };
    GenericTensorAccessorW tensor_backing =
        allocator.allocate_tensor(input_tensor_shape);

    // build graph
    ComputationGraphBuilder cg_builder;
    TensorBackingMap tensor_backing_map;
    tensor_guid_t input_tensor_guid =
        cg_builder.create_tensor(input_tensor_shape, CreateGrad::YES);
    tensor_backing_map.insert({input_tensor_guid, tensor_backing});
    tensor_guid_t output_tensor_guid =
        cg_builder.cast(input_tensor_guid, output_dtype);

    // local backing initialization
    RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
        DeviceSpecific<PerDeviceFFHandle>::create(get_per_device_ff_handle()),
        EnableProfiling::NO,
        ProfilingSettings{/*warmup_iters=*/0,
                          /*measure_iters=*/0}};
    LocalTrainingBacking local_backing =
        LocalTrainingBacking{allocator,
                             cg_builder.computation_graph,
                             tensor_backing_map,
                             runtime_arg_config};

    layer_guid_t layer_guid =
        get_only(topological_ordering(cg_builder.computation_graph));
    TaskRegistry task_registry = local_backing.get_task_registry();
    LocalSlotsBacking local_slots_backing =
        local_backing.get_local_slots_backing();

    SUBCASE("Task Registration") {
      SUBCASE("Task Registration Fwd") {
        std::unordered_map<layer_guid_t, task_id_t> correct_forward_task_ids = {
            {layer_guid, CAST_FWD_TASK_ID}};
        CHECK(correct_forward_task_ids == task_registry.forward_task_ids);
      }

      SUBCASE("Task Registration Bwd") {
        std::unordered_map<layer_guid_t, task_id_t> correct_backward_task_ids =
            {{layer_guid, CAST_BWD_TASK_ID}};
        CHECK(correct_backward_task_ids == task_registry.backward_task_ids);
      }
    }

    SUBCASE("Tensor Slot Insertion") {
      SUBCASE("Input Tensor Slot Insertion") {
        std::unordered_map<layer_guid_t, std::vector<tensor_guid_t>>
            correct_input_tensor_slots = {
                {layer_guid, std::vector<tensor_guid_t>{input_tensor_guid}}};
        CHECK(correct_input_tensor_slots ==
              local_slots_backing.input_tensor_slots);
      }

      SUBCASE("Output Tensor Slot Insertion") {
        std::unordered_map<layer_guid_t, std::vector<tensor_guid_t>>
            correct_output_tensor_slots = {
                {layer_guid, std::vector<tensor_guid_t>{output_tensor_guid}}};
        CHECK(correct_output_tensor_slots ==
              local_slots_backing.output_tensor_slots);
      }
    }

    SUBCASE("Tensor Allocation") {
      SUBCASE("Input Tensor Allocation") {
        GenericTensorAccessorW input_tensor_backing =
            local_slots_backing.tensor_mapping.at(input_tensor_guid);
        CHECK(is_shape_and_dtype_correct(
            input_tensor_backing, ArrayShape{input_tensor_shape}, input_dtype));
      }

      SUBCASE("Output Tensor Allocation") {
        GenericTensorAccessorW output_tensor_backing =
            local_slots_backing.tensor_mapping.at(output_tensor_guid);
        CHECK(is_shape_and_dtype_correct(output_tensor_backing,
                                         ArrayShape{input_tensor_shape},
                                         output_dtype));
      }

      SUBCASE("Input Gradient Tensor Allocation") {
        GenericTensorAccessorW input_grad_tensor_backing =
            local_slots_backing.gradient_tensor_mapping.at(input_tensor_guid);
        CHECK(is_shape_and_dtype_correct(input_grad_tensor_backing,
                                         ArrayShape{input_tensor_shape},
                                         input_dtype));
      }

      SUBCASE("Output Gradient Tensor Allocation") {
        GenericTensorAccessorW output_grad_tensor_backing =
            local_slots_backing.gradient_tensor_mapping.at(output_tensor_guid);
        CHECK(is_shape_and_dtype_correct(output_grad_tensor_backing,
                                         ArrayShape{input_tensor_shape},
                                         output_dtype));
      }
    }

    SUBCASE("Task Argument Accessor") {
      enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };
      CastAttrs attrs = CastAttrs{output_dtype};
      OpTaskBinding binding = [&] {
        OpTaskBinding b;
        b.bind_arg(PROFILING, profiling_settings());
        b.bind_arg(ATTRS, attrs);
        b.bind(INPUT, input_tensor(0));
        b.bind(OUTPUT, output_tensor(0));
        return b;
      }();

      SUBCASE("Forward Task Argument Accessor") {
        OpTaskInvocation fwd_invocation = {CAST_FWD_TASK_ID, binding};
        TaskArgumentAccessor acc =
            local_backing.get_task_arg_accessor(fwd_invocation, layer_guid);

        SUBCASE("Profiling Settings") {
          ProfilingSettings result_settings =
              acc.get_argument<ProfilingSettings>(PROFILING);
          CHECK(runtime_arg_config.profiling_settings == result_settings);
        }
        SUBCASE("Attrs") {
          CastAttrs result_attrs = acc.get_argument<CastAttrs>(ATTRS);
          CHECK(attrs == result_attrs);
        }
        SUBCASE("Matches Tensors from Slots Backing") {
          SUBCASE("Input Tensor") {
            GenericTensorAccessorW result_input_tensor_backing =
                acc.get_tensor<Permissions::WO>(INPUT);
            GenericTensorAccessorW slots_input_tensor_backing =
                local_slots_backing.tensor_mapping.at(input_tensor_guid);
            CHECK(slots_input_tensor_backing == result_input_tensor_backing);
          }
          SUBCASE("Output Tensor") {
            GenericTensorAccessorW result_output_tensor_backing =
                acc.get_tensor<Permissions::WO>(OUTPUT);
            GenericTensorAccessorW slots_output_tensor_backing =
                local_slots_backing.tensor_mapping.at(output_tensor_guid);
            CHECK(slots_output_tensor_backing == result_output_tensor_backing);
          }
        }
      }

      SUBCASE("Backward Task Argument Accessor") {
        binding.bind_grad(INPUT, input_tensor(0));
        binding.bind_grad(OUTPUT, output_tensor(0));
        OpTaskInvocation bwd_invocation = {CAST_BWD_TASK_ID, binding};
        TaskArgumentAccessor acc =
            local_backing.get_task_arg_accessor(bwd_invocation, layer_guid);

        SUBCASE("Matches Grad Tensors from Slots Backing") {
          SUBCASE("Input Grad Tensor") {
            GenericTensorAccessorW result_input_grad_tensor_backing =
                acc.get_tensor<Permissions::WO>(INPUT);
            GenericTensorAccessorW slots_input_grad_tensor_backing =
                local_slots_backing.gradient_tensor_mapping.at(
                    input_tensor_guid);
            CHECK(slots_input_grad_tensor_backing ==
                  result_input_grad_tensor_backing);
          }
          SUBCASE("Output Grad Tensor") {
            GenericTensorAccessorW result_output_grad_tensor_backing =
                acc.get_tensor<Permissions::WO>(OUTPUT);
            GenericTensorAccessorW slots_output_grad_tensor_backing =
                local_slots_backing.gradient_tensor_mapping.at(
                    output_tensor_guid);
            CHECK(slots_output_grad_tensor_backing ==
                  result_output_grad_tensor_backing);
          }
        }
      }
    }

    SUBCASE("End-to-end") {
      local_backing.execute_init();
      local_backing.execute_forward();
      local_backing.execute_backward();
    }
  }
}
} // namespace FlexFlow
