#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/test/src/test_utils.h"
#include "local-execution/local_cost_estimator.h"
#include "pcg/computation_graph_builder.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Local Backing Multiple Operators -- Cast->Linear") {
    Allocator allocator = get_local_memory_allocator();

    // define operators
    size_t batch_size = 12;
    size_t extra_dim = 16;
    size_t in_channels = 8;

    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                batch_size,
                extra_dim,
                in_channels,
            },
        },
        DataType::FLOAT,
    };
    // -- cast
    DataType input_datatype = DataType::FLOAT;
    DataType output_datatype = DataType::DOUBLE;
    TensorAttrs cast_output = TensorAttrs{TensorShape{
                                              input.dims,
                                              DataType::DOUBLE,
                                          },
                                          std::nullopt,
                                          std::nullopt,
                                          CreateGrad::YES};
    CastAttrs cast_attrs = CastAttrs{output_datatype};

    // -- linear
    int out_channels = 16;
    TensorAttrs linear_output_shape =
        TensorAttrs{TensorShape{
                        TensorDims{
                            FFOrdered<size_t>{
                                batch_size,
                                extra_dim,
                                size_t_from_int(out_channels),
                            },
                        },
                        DataType::FLOAT,
                    },
                    std::nullopt,
                    std::nullopt,
                    CreateGrad::YES};
    TensorAttrs kernel = TensorAttrs{TensorShape{
                                         TensorDims{
                                             FFOrdered<size_t>{
                                                 in_channels,
                                                 size_t_from_int(out_channels),
                                             },
                                         },
                                         DataType::DOUBLE,
                                     },
                                     std::nullopt,
                                     std::nullopt,
                                     CreateGrad::YES};
    TensorAttrs bias = TensorAttrs{TensorShape{
                                       TensorDims{
                                           FFOrdered<size_t>{
                                               size_t_from_int(out_channels),
                                           },
                                       },
                                       DataType::DOUBLE,
                                   },
                                   std::nullopt,
                                   std::nullopt,
                                   CreateGrad::NO};
    LinearAttrs linear_attrs = LinearAttrs{/*out_channels=*/out_channels,
                                           /*use_bias=*/true,
                                           /*data_type=*/DataType::DOUBLE,
                                           /*activation=*/Activation::RELU,
                                           /*regularizer=*/std::nullopt};

    // build graph
    TensorBackingMap tensor_backing_map;
    ComputationGraphBuilder cg_builder;
    tensor_guid_t tensor_id = cg_builder.create_tensor(input, CreateGrad::YES);
    GenericTensorAccessorW tensor_backing = allocator.allocate_tensor(input);
    tensor_backing_map.insert({tensor_id, tensor_backing});
    std::vector<tensor_guid_t> cast_result = cg_builder.add_layer(
        LayerAttrs{ComputationGraphOpAttrs{cast_attrs}, std::nullopt},
        std::vector<tensor_guid_t>{tensor_id},
        {},
        std::vector<TensorAttrs>{cast_output});
    cg_builder.add_layer(
        LayerAttrs{ComputationGraphOpAttrs{linear_attrs}, std::nullopt},
        cast_result,
        std::vector<TensorAttrs>{kernel, bias},
        std::vector<TensorAttrs>{linear_output_shape});

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

    std::vector<layer_guid_t> layer_guids =
        topological_ordering(cg_builder.computation_graph);
    layer_guid_t cast_layer_guid = layer_guids.at(0);
    layer_guid_t linear_layer_guid = layer_guids.at(1);

    SUBCASE("Task Registration") {
      TaskRegistry const task_registry = local_backing.get_task_registry();
      std::unordered_map<layer_guid_t, task_id_t> correct_init_task_ids;
      std::unordered_map<layer_guid_t, task_id_t> correct_forward_task_ids;
      std::unordered_map<layer_guid_t, task_id_t> correct_backward_task_ids;

      correct_forward_task_ids.insert({cast_layer_guid, CAST_FWD_TASK_ID});
      correct_backward_task_ids.insert({cast_layer_guid, CAST_BWD_TASK_ID});

      correct_init_task_ids.insert({linear_layer_guid, LINEAR_INIT_TASK_ID});
      correct_forward_task_ids.insert({linear_layer_guid, LINEAR_FWD_TASK_ID});
      correct_backward_task_ids.insert({linear_layer_guid, LINEAR_BWD_TASK_ID});

      CHECK(correct_init_task_ids == task_registry.init_task_ids);
      CHECK(correct_forward_task_ids == task_registry.forward_task_ids);
      CHECK(correct_backward_task_ids == task_registry.backward_task_ids);
    }

    std::vector<tensor_guid_t> cast_input_tensor_guids =
        get_incoming_tensors(cg_builder.computation_graph, cast_layer_guid);
    std::vector<tensor_guid_t> cast_output_tensor_guids =
        get_outgoing_tensors(cg_builder.computation_graph, cast_layer_guid);
    std::vector<tensor_guid_t> linear_input_tensor_guids =
        get_incoming_tensors(cg_builder.computation_graph, linear_layer_guid);
    std::vector<tensor_guid_t> linear_output_tensor_guids =
        get_outgoing_tensors(cg_builder.computation_graph, linear_layer_guid);

    SUBCASE("Tensor Slot Insertion") {
      LocalSlotsBacking const local_slots_backing =
          local_backing.get_local_slots_backing();
      std::unordered_map<layer_guid_t, std::vector<tensor_guid_t>>
          correct_input_tensor_slots;
      std::unordered_map<layer_guid_t, std::vector<tensor_guid_t>>
          correct_output_tensor_slots;

      correct_input_tensor_slots.insert(
          {cast_layer_guid, cast_input_tensor_guids});
      correct_output_tensor_slots.insert(
          {cast_layer_guid, cast_output_tensor_guids});
      correct_input_tensor_slots.insert(
          {linear_layer_guid, linear_input_tensor_guids});
      correct_output_tensor_slots.insert(
          {linear_layer_guid, linear_output_tensor_guids});

      CHECK(correct_input_tensor_slots ==
            local_slots_backing.input_tensor_slots);
      CHECK(correct_output_tensor_slots ==
            local_slots_backing.output_tensor_slots);
    }

    LocalSlotsBacking const local_slots_backing =
        local_backing.get_local_slots_backing();

    SUBCASE("Linear Tensor Allocation") {
      tensor_guid_t linear_input = linear_input_tensor_guids.at(0);
      tensor_guid_t linear_kernel = linear_input_tensor_guids.at(1);
      tensor_guid_t linear_bias = linear_input_tensor_guids.at(2);
      tensor_guid_t linear_output = linear_output_tensor_guids.at(0);

      GenericTensorAccessorW input_tensor_backing =
          local_slots_backing.tensor_mapping.at(linear_input);
      GenericTensorAccessorW kernel_tensor_backing =
          local_slots_backing.tensor_mapping.at(linear_kernel);
      GenericTensorAccessorW bias_tensor_backing =
          local_slots_backing.tensor_mapping.at(linear_bias);
      GenericTensorAccessorW output_tensor_backing =
          local_slots_backing.tensor_mapping.at(linear_output);

      CHECK(ArrayShape{cast_output.shape} == input_tensor_backing.shape);
      CHECK(output_datatype == input_tensor_backing.data_type);
      CHECK(ArrayShape{kernel.shape} == kernel_tensor_backing.shape);
      CHECK(ArrayShape{bias.shape} == bias_tensor_backing.shape);
      CHECK(ArrayShape{linear_output_shape.shape} ==
            output_tensor_backing.shape);

      // gradient tensors
      GenericTensorAccessorW input_grad_tensor_backing =
          local_slots_backing.gradient_tensor_mapping.at(linear_input);
      GenericTensorAccessorW kernel_grad_tensor_backing =
          local_slots_backing.gradient_tensor_mapping.at(linear_kernel);
      GenericTensorAccessorW output_grad_tensor_backing =
          local_slots_backing.gradient_tensor_mapping.at(linear_output);

      CHECK(ArrayShape{cast_output.shape} == input_grad_tensor_backing.shape);
      CHECK(output_datatype == input_tensor_backing.data_type);
      CHECK(ArrayShape{kernel.shape} == kernel_grad_tensor_backing.shape);
      CHECK(ArrayShape{linear_output_shape.shape} ==
            output_grad_tensor_backing.shape);
    }

    SUBCASE("Per Device State") {
      local_backing.execute_init();
      CHECK(!contains_key(local_slots_backing.per_device_op_states,
                          cast_layer_guid));
      CHECK(contains_key(local_slots_backing.per_device_op_states,
                         linear_layer_guid));
    }

    SUBCASE("End-to-end") {
      local_backing.execute_init();
      local_backing.execute_forward();
      local_backing.execute_backward();
    }
  }
}
} // namespace FlexFlow
