#include "doctest/doctest.h"
#include "local-execution/local_cost_estimator.h"
#include "local-execution/local_cpu_allocator.h"
#include "local-execution/local_slots_backing.h"
#include "pcg/computation_graph_builder.h"
#include "test_utils.h"

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Local Slots Backing -- Attention Op") {
    // allocate input memory
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
    GenericTensorAccessorW key = allocator.allocate_tensor(input_tensor_shape);
    GenericTensorAccessorW value =
        allocator.allocate_tensor(input_tensor_shape);

    // build graph
    ComputationGraphBuilder cg_builder;
    tensor_guid_t query_guid =
        cg_builder.create_tensor(input_tensor_shape, CreateGrad::YES);
    tensor_guid_t key_guid =
        cg_builder.create_tensor(input_tensor_shape, CreateGrad::YES);
    tensor_guid_t value_guid =
        cg_builder.create_tensor(input_tensor_shape, CreateGrad::YES);
    tensor_guid_t output_guid =
        cg_builder.multihead_attention(query_guid,
                                       key_guid,
                                       value_guid,
                                       embed_dim,
                                       num_heads,
                                       /*kdim=*/embed_dim,
                                       /*vdim=*/embed_dim);
    // TODO: @lockshaw replace with named layers
    layer_guid_t layer_guid =
        get_only(topological_ordering(cg_builder.computation_graph));

    TensorBackingMap tensor_backing_map = {
        {query_guid, query}, {key_guid, key}, {value_guid, value}};

    // runtime arg config
    ProfilingSettings settings = ProfilingSettings{/*warmup_iters=*/0,
                                                   /*measure_iters=*/0};
    RuntimeArgConfig runtime_arg_config =
        RuntimeArgConfig{DeviceSpecific<PerDeviceFFHandle>::create(
                             get_mock_per_device_ff_handle()),
                         EnableProfiling::NO,
                         settings};

    LocalSlotsBacking local_slots_backing = {tensor_backing_map,
                                             runtime_arg_config};

    SUBCASE("Allocate and insert new tensors into slots") {
      for (layer_guid_t const &node :
           topological_ordering(cg_builder.computation_graph)) {
        local_slots_backing.allocate_new_tensors(
            node, cg_builder.computation_graph, allocator);
      }

      SUBCASE("Tensor allocation") {
        SUBCASE("Query grad") {
          GenericTensorAccessorW query_grad =
              local_slots_backing.gradient_tensor_mapping.at(query_guid);
          CHECK(is_shape_and_dtype_correct(
              query_grad, ArrayShape{input_tensor_shape}, dtype));
        }
        SUBCASE("Key grad") {
          GenericTensorAccessorW key_grad =
              local_slots_backing.gradient_tensor_mapping.at(key_guid);
          CHECK(is_shape_and_dtype_correct(
              key_grad, ArrayShape{input_tensor_shape}, dtype));
        }
        SUBCASE("Value grad") {
          GenericTensorAccessorW value_grad =
              local_slots_backing.gradient_tensor_mapping.at(value_guid);
          CHECK(is_shape_and_dtype_correct(
              value_grad, ArrayShape{input_tensor_shape}, dtype));
        }

        TensorAttrs output_attrs =
            get_tensor_attrs(cg_builder.computation_graph, output_guid);
        SUBCASE("Output") {
          GenericTensorAccessorW output =
              local_slots_backing.tensor_mapping.at(output_guid);
          CHECK(is_shape_and_dtype_correct(
              output, ArrayShape{output_attrs.shape}, dtype));
        }
        SUBCASE("Output grad") {
          GenericTensorAccessorW output_grad =
              local_slots_backing.tensor_mapping.at(output_guid);
          CHECK(is_shape_and_dtype_correct(
              output_grad, ArrayShape{output_attrs.shape}, dtype));
        }
      }

      SUBCASE("Tensor slots") {
        SUBCASE("Input tensor slots") {
          std::vector<tensor_guid_t> correct_incoming_tensors =
              get_incoming_tensors(cg_builder.computation_graph, layer_guid);
          CHECK(correct_incoming_tensors ==
                local_slots_backing.input_tensor_slots.at(layer_guid));
        }
        SUBCASE("Output tensor slots") {
          std::vector<tensor_guid_t> correct_outgoing_tensors =
              get_outgoing_tensors(cg_builder.computation_graph, layer_guid);
          CHECK(correct_outgoing_tensors ==
                local_slots_backing.output_tensor_slots.at(layer_guid));
        }
      }
    }

    SUBCASE("Construct Slots Backings -- Mock Binding") {
      enum Slots {
        QUERY,
        KEY,
        VALUE,
        WEIGHTS,
        OUTPUT,
        QUERY_PARALLEL_TENSOR_SHAPE,
        QPROJSIZE,
        ATTRS,
        PROFILING,
        HANDLE,
      };
      MultiHeadAttentionAttrs attrs =
          get_layer_attrs(cg_builder.computation_graph, layer_guid)
              .attrs.get<MultiHeadAttentionAttrs>();
      OpTaskBinding mock_binding = [&] {
        // mock binding to avoid running any kernels
        OpTaskBinding b;
        b.bind(QUERY, input_tensor(0));
        b.bind(KEY, input_tensor(1));
        b.bind(VALUE, input_tensor(2));
        b.bind(WEIGHTS, weight_tensor(3));
        b.bind(OUTPUT, output_tensor(0));

        b.bind_grad(QUERY, input_tensor(0));

        b.bind_arg(QPROJSIZE, get_qProjSize(attrs));
        b.bind_arg(ATTRS, attrs);
        b.bind_arg(QUERY_PARALLEL_TENSOR_SHAPE, input_parallel_tensor_shape(0));
        b.bind_arg(PROFILING, profiling_settings());
        b.bind_arg(HANDLE, ff_handle());
        return b;
      }();

      SUBCASE("Tensor Slots Backing") {
        TensorSlotsBacking result =
            local_slots_backing.construct_tensor_slots_backing(mock_binding,
                                                               layer_guid);

        TensorShape weights_shape = throw_if_unexpected(get_weights_shape(
            attrs, input_tensor_shape, input_tensor_shape, input_tensor_shape));
        GenericTensorAccessorW weights =
            allocator.allocate_tensor(weights_shape);

        TensorAttrs output_attrs =
            get_tensor_attrs(cg_builder.computation_graph, output_guid);
        GenericTensorAccessorW output =
            allocator.allocate_tensor(output_attrs.shape);
        TensorSlotsBacking correct = {{{QUERY, IsGrad::NO}, query},
                                      {{KEY, IsGrad::NO}, weights},
                                      {{VALUE, IsGrad::NO}, weights},
                                      {{WEIGHTS, IsGrad::NO}, weights},
                                      {{OUTPUT, IsGrad::NO}, output},
                                      {{QUERY, IsGrad::YES}, query}};

        CHECK(are_slots_backings_virtually_equivalent(correct, result));
      }
      SUBCASE("Arg Slots Backing") {
        ArgSlotsBacking result =
            local_slots_backing.construct_arg_slots_backing(mock_binding,
                                                            layer_guid);

        ParallelTensorShape query_parallel_tensor_shape =
            lift_to_parallel(input_tensor_shape);
        ArgSlotsBacking correct = {
            {QPROJSIZE, ConcreteArgSpec::create(get_qProjSize(attrs))},
            {ATTRS, ConcreteArgSpec::create(attrs)},
            {QUERY_PARALLEL_TENSOR_SHAPE,
             ConcreteArgSpec::create(query_parallel_tensor_shape)},
            {PROFILING,
             ConcreteArgSpec::create(runtime_arg_config.profiling_settings)},
            {HANDLE, ConcreteArgSpec::create(runtime_arg_config.ff_handle)}};

        CHECK(result == correct);
      }
    }
  }
}

} // namespace FlexFlow
