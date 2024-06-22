#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/test/src/test_utils.h"
#include "local-execution/local_cost_estimator.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Local Backing Single Operator -- Cast") {
    Allocator allocator = get_local_memory_allocator();

    // define operator
    size_t d1 = 12;
    size_t d2 = 16;
    DataType input_datatype = DataType::FLOAT;
    DataType output_datatype = DataType::DOUBLE;

    TensorShape input = TensorShape{
        TensorDims{FFOrdered<size_t>{d1, d2}},
        input_datatype,
    };
    TensorAttrs output = TensorAttrs{TensorShape{
                                         TensorDims{FFOrdered<size_t>{d1, d2}},
                                         output_datatype,
                                     },
                                     std::nullopt,
                                     std::nullopt,
                                     CreateGrad::YES};
    CastAttrs attrs = CastAttrs{output_datatype};

    // build graph
    TensorBackingMap tensor_backing_map;
    ComputationGraphBuilder cg_builder;
    tensor_guid_t tensor_id = cg_builder.create_tensor(input, CreateGrad::YES);
    GenericTensorAccessorW tensor_backing = allocator.allocate_tensor(input);
    tensor_backing_map.insert({tensor_id, tensor_backing});
    cg_builder.add_layer(
        LayerAttrs{ComputationGraphOpAttrs{attrs}, std::nullopt},
        std::vector<tensor_guid_t>{tensor_id},
        {},
        std::vector<TensorAttrs>{output});

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

    SUBCASE("Task Registration") {
      TaskRegistry const task_registry = local_backing.get_task_registry();
      std::unordered_map<layer_guid_t, task_id_t> correct_forward_task_ids;
      std::unordered_map<layer_guid_t, task_id_t> correct_backward_task_ids;

      correct_forward_task_ids.insert({layer_guid, CAST_FWD_TASK_ID});
      correct_backward_task_ids.insert({layer_guid, CAST_BWD_TASK_ID});

      CHECK(correct_forward_task_ids == task_registry.forward_task_ids);
      CHECK(correct_backward_task_ids == task_registry.backward_task_ids);
    }

    std::vector<tensor_guid_t> input_tensor_guids =
        get_incoming_tensors(cg_builder.computation_graph, layer_guid);
    std::vector<tensor_guid_t> output_tensor_guids =
        get_outgoing_tensors(cg_builder.computation_graph, layer_guid);

    SUBCASE("Tensor Slot Insertion") {
      LocalSlotsBacking const local_slots_backing =
          local_backing.get_local_slots_backing();
      std::unordered_map<layer_guid_t, std::vector<tensor_guid_t>>
          correct_input_tensor_slots;
      std::unordered_map<layer_guid_t, std::vector<tensor_guid_t>>
          correct_output_tensor_slots;

      correct_input_tensor_slots.insert({layer_guid, input_tensor_guids});
      correct_output_tensor_slots.insert({layer_guid, output_tensor_guids});

      CHECK(correct_input_tensor_slots ==
            local_slots_backing.input_tensor_slots);
      CHECK(correct_output_tensor_slots ==
            local_slots_backing.output_tensor_slots);
    }

    ArrayShape correct_input_array_shape = ArrayShape{input};
    ArrayShape correct_output_array_shape = ArrayShape{output.shape};
    tensor_guid_t input_tensor_guid = get_only(input_tensor_guids);
    tensor_guid_t output_tensor_guid = get_only(output_tensor_guids);
    LocalSlotsBacking const local_slots_backing =
        local_backing.get_local_slots_backing();

    SUBCASE("Tensor Allocation") {
      GenericTensorAccessorW input_tensor_backing =
          local_slots_backing.tensor_mapping.at(input_tensor_guid);
      GenericTensorAccessorW output_tensor_backing =
          local_slots_backing.tensor_mapping.at(output_tensor_guid);
      CHECK(correct_input_array_shape == input_tensor_backing.shape);
      CHECK(input_datatype == input_tensor_backing.data_type);
      CHECK(correct_output_array_shape == output_tensor_backing.shape);
      CHECK(output_datatype == output_tensor_backing.data_type);

      // gradient tensors
      GenericTensorAccessorW input_grad_tensor_backing =
          local_slots_backing.gradient_tensor_mapping.at(input_tensor_guid);
      GenericTensorAccessorW output_grad_tensor_backing =
          local_slots_backing.gradient_tensor_mapping.at(output_tensor_guid);
      CHECK(correct_input_array_shape == input_grad_tensor_backing.shape);
      CHECK(input_datatype == input_grad_tensor_backing.data_type);
      CHECK(correct_output_array_shape == output_grad_tensor_backing.shape);
      CHECK(output_datatype == output_grad_tensor_backing.data_type);
    }

    enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };
    SUBCASE("Forward Task Argument Accessor") {
      // initialize invocation
      OpTaskBinding binding;
      binding.bind_arg(PROFILING, profiling_settings());
      binding.bind_arg(ATTRS, attrs);
      binding.bind(INPUT, input_tensor(0));
      binding.bind(OUTPUT, output_tensor(0));
      OpTaskInvocation fwd_invocation = {CAST_FWD_TASK_ID, binding};
      ProfilingSettings correct_settings = ProfilingSettings{0, 0};

      // get accessor
      TaskArgumentAccessor acc =
          local_backing.get_task_arg_accessor(fwd_invocation, layer_guid);
      ProfilingSettings result_settings =
          acc.get_argument<ProfilingSettings>(PROFILING);
      CastAttrs result_attrs = acc.get_argument<CastAttrs>(ATTRS);
      GenericTensorAccessorR result_input_tensor_backing =
          acc.get_tensor<Permissions::RO>(INPUT);
      GenericTensorAccessorW result_output_tensor_backing =
          acc.get_tensor<Permissions::WO>(OUTPUT);

      CHECK(correct_settings == result_settings);
      CHECK(attrs == result_attrs);
      CHECK(correct_input_array_shape == result_input_tensor_backing.shape);
      CHECK(input_datatype == result_input_tensor_backing.data_type);
      CHECK(correct_output_array_shape == result_output_tensor_backing.shape);
      CHECK(output_datatype == result_output_tensor_backing.data_type);

      // check against slots backing
      LocalSlotsBacking const local_slots_backing =
          local_backing.get_local_slots_backing();
      GenericTensorAccessorW slots_input_tensor_backing =
          local_slots_backing.tensor_mapping.at(input_tensor_guid);
      GenericTensorAccessorW slots_output_tensor_backing =
          local_slots_backing.tensor_mapping.at(output_tensor_guid);
      CHECK(result_input_tensor_backing ==
            read_only_accessor_from_write_accessor(slots_input_tensor_backing));
      CHECK(result_output_tensor_backing == slots_output_tensor_backing);
    }

    SUBCASE("Backward Task Argument Accessor") {
      // initialize invocation
      OpTaskBinding binding;
      binding.bind_arg(PROFILING, profiling_settings());
      binding.bind_arg(ATTRS, attrs);
      binding.bind(INPUT, input_tensor(0));
      binding.bind_grad(INPUT, input_tensor(0));
      binding.bind(OUTPUT, output_tensor(0));
      binding.bind_grad(OUTPUT, output_tensor(0));
      OpTaskInvocation bwd_invocation = {CAST_BWD_TASK_ID, binding};
      ProfilingSettings correct_settings = ProfilingSettings{0, 0};

      // get acc
      TaskArgumentAccessor acc =
          local_backing.get_task_arg_accessor(bwd_invocation, layer_guid);
      ProfilingSettings result_settings =
          acc.get_argument<ProfilingSettings>(PROFILING);
      CastAttrs result_attrs = acc.get_argument<CastAttrs>(ATTRS);
      GenericTensorAccessorR result_input_tensor_backing =
          acc.get_tensor<Permissions::RO>(INPUT);
      GenericTensorAccessorR result_input_grad_tensor_backing =
          acc.get_tensor_grad<Permissions::RO>(INPUT);
      GenericTensorAccessorW result_output_grad_tensor_backing =
          acc.get_tensor_grad<Permissions::WO>(OUTPUT);

      CHECK(correct_settings == result_settings);
      CHECK(attrs == result_attrs);
      CHECK(correct_input_array_shape == result_input_tensor_backing.shape);
      CHECK(correct_input_array_shape ==
            result_input_grad_tensor_backing.shape);
      CHECK(input_datatype == result_input_tensor_backing.data_type);
      CHECK(input_datatype == result_input_grad_tensor_backing.data_type);
      CHECK(correct_output_array_shape ==
            result_output_grad_tensor_backing.shape);
      CHECK(output_datatype == result_output_grad_tensor_backing.data_type);

      // check against slots backing
      LocalSlotsBacking const local_slots_backing =
          local_backing.get_local_slots_backing();
      GenericTensorAccessorW slots_input_tensor_backing =
          local_slots_backing.tensor_mapping.at(input_tensor_guid);
      GenericTensorAccessorW slots_input_grad_tensor_backing =
          local_slots_backing.gradient_tensor_mapping.at(input_tensor_guid);
      GenericTensorAccessorW slots_output_grad_tensor_backing =
          local_slots_backing.gradient_tensor_mapping.at(output_tensor_guid);

      CHECK(result_input_tensor_backing ==
            read_only_accessor_from_write_accessor(slots_input_tensor_backing));
      CHECK(result_input_grad_tensor_backing ==
            read_only_accessor_from_write_accessor(
                slots_input_grad_tensor_backing));
      CHECK(result_output_grad_tensor_backing ==
            slots_output_grad_tensor_backing);
    }

    SUBCASE("End-to-end") {
      local_backing.execute_init();
      local_backing.execute_forward();
      local_backing.execute_backward();
    }
  }
}
} // namespace FlexFlow
