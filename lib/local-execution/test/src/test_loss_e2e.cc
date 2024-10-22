#include "doctest/doctest.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/local_training_backing.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "test_utils.h"

namespace FlexFlow {

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Local Execution E2E") {
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
    std::string layer_name = "scalar multiply";
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
    local_backing.register_and_allocate_layer(
        get_layer_by_name(cg_builder.computation_graph, layer_name));

    SUBCASE("SparseCategoricalCrossEntropyLossAttrs") {
      TensorShape label_shape = TensorShape{
          TensorDims{FFOrdered<size_t>{batch_size, 1}}, DataType::FLOAT};
      tensor_guid_t label_tensor =
          cg_builder.create_input(label_shape, CreateGrad::NO);
      GenericTensorAccessorW label_backing =
          allocator.allocate_tensor(label_shape);
      local_backing.insert_tensor(label_tensor, label_backing);
      LossAttrs loss_attrs = LossAttrs{
          SparseCategoricalCrossEntropyLossAttrs{/*replace_labels=*/false}};
      local_backing.compute_loss(loss_attrs, logit_tensor, label_tensor);
    }

    SUBCASE("NonconfigurableLossAttrs") {
      tensor_guid_t label_tensor =
          cg_builder.create_input(input_shape, CreateGrad::NO);
      GenericTensorAccessorW label_backing =
          allocator.allocate_tensor(input_shape);
      local_backing.insert_tensor(label_tensor, label_backing);

      SUBCASE("LossFunction::CATEGORICAL_CROSSENTROPY") {
        LossAttrs loss_attrs = LossAttrs{
            NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
        local_backing.compute_loss(loss_attrs, logit_tensor, label_tensor);
      }

      SUBCASE("LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE") {
        LossAttrs loss_attrs = LossAttrs{NonconfigurableLossAttrs{
            LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE}};
        local_backing.compute_loss(loss_attrs, logit_tensor, label_tensor);
      }

      SUBCASE("LossFunction::IDENTITY") {
        LossAttrs loss_attrs =
            LossAttrs{NonconfigurableLossAttrs{LossFunction::IDENTITY}};
        local_backing.compute_loss(loss_attrs, logit_tensor, label_tensor);
      }
    }
  }
}

} // namespace FlexFlow
