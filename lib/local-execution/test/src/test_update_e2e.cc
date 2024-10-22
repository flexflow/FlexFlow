#include "doctest/doctest.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/local_training_backing.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "test_utils.h"

namespace FlexFlow {

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Local Execution Update E2E") {
    // initialize runtime configs
    ManagedPerDeviceFFHandle managed_handle{};

    RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
        DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
        EnableProfiling::YES,
        ProfilingSettings{/*warmup_iters=*/0, /*measure_iters=*/1}};

    // construct graph
    ComputationGraphBuilder cg_builder;

    size_t batch_size = 10;
    size_t data_dim = 100;
    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered<size_t>{batch_size, data_dim}}, DataType::FLOAT};
    tensor_guid_t input_tensor =
        cg_builder.create_input(input_shape, CreateGrad::YES);

    float scalar = 4.0;
    std::string layer_name = "scalar_multiply";
    tensor_guid_t logit_tensor =
        cg_builder.scalar_multiply(input_tensor, scalar, layer_name);

    // allocate memory
    Allocator allocator = create_local_cuda_memory_allocator();
    TensorBackingMap tensor_backing_map;
    GenericTensorAccessorW input_backing =
        allocator.allocate_tensor(input_shape);
    tensor_backing_map.insert({input_tensor, input_backing});

    LocalTrainingBacking local_backing(allocator,
                                       cg_builder.computation_graph,
                                       tensor_backing_map,
                                       runtime_arg_config);
    // for (layer_guid_t const & node:
    // topological_ordering(cg_builder.computation_graph)) {
    //   local_backing.register_and_allocate_layer(node);
    // }
    layer_guid_t layer_guid =
        get_layer_by_name(cg_builder.computation_graph, layer_name);
    local_backing.register_and_allocate_layer(layer_guid);

    SUBCASE("SGDOptimizerAttrs") {
      SUBCASE("momentum=0") {
        OptimizerAttrs optimizer_attrs =
            OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                             /*momentum=*/0.0f,
                                             /*nesterov=*/false,
                                             /*weight_decay=*/0.001}};
        local_backing.allocate_layer_optimizer_tensors(layer_guid,
                                                       optimizer_attrs);
        local_backing.execute_update(layer_guid, optimizer_attrs);
      }
      SUBCASE("momentum=0.9") {
        OptimizerAttrs optimizer_attrs =
            OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                             /*momentum=*/0.9,
                                             /*nesterov=*/false,
                                             /*weight_decay=*/0.001}};
        local_backing.allocate_layer_optimizer_tensors(layer_guid,
                                                       optimizer_attrs);
        local_backing.execute_update(layer_guid, optimizer_attrs);
      }
    }
    SUBCASE("AdamOptimizerAttrs") {
      OptimizerAttrs optimizer_attrs =
          OptimizerAttrs{AdamOptimizerAttrs{/*alpha=*/0.001,
                                            /*beta1=*/0.9,
                                            /*beta2=*/0.999,
                                            /*weight_decay=*/0.001,
                                            /*alpha_t=*/0.001,
                                            /*beta_t=*/0.9,
                                            /*beta2_t=*/0.999,
                                            /*epsilon=*/1e-8}};
      local_backing.allocate_layer_optimizer_tensors(layer_guid,
                                                     optimizer_attrs);
      local_backing.execute_update(layer_guid, optimizer_attrs);
    }
  }
}

} // namespace FlexFlow
