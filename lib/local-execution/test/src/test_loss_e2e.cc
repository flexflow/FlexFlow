#include "doctest/doctest.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/local_training_backing.h"
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

    OptimizerAttrs optimizer_attrs =
        OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.0,
                                         /*momentum=*/0.0,
                                         /*nesterov=*/false,
                                         /*weight_decay=*/0.0}};

    // construct graph
    ComputationGraphBuilder cg_builder;

    size_t batch_size = 10;
    size_t data_dim = 100;
    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered<size_t>{batch_size, data_dim}}, DataType::FLOAT};
    tensor_guid_t input_tensor =
        cg_builder.create_input(input_shape, CreateGrad::YES);

    float scalar = 4.0;
    tensor_guid_t logit_tensor =
        cg_builder.scalar_multiply(input_tensor, scalar);

    // allocate memory
    Allocator allocator = create_local_cuda_memory_allocator();
    TensorBackingMap tensor_backing_map;
    GenericTensorAccessorW input_backing =
        allocator.allocate_tensor(input_shape);
    tensor_backing_map.insert({input_tensor, input_backing});

    SUBCASE("SparseCategoricalCrossEntropyLossAttrs") {
      TensorShape label_shape = TensorShape{
          TensorDims{FFOrdered<size_t>{batch_size, 1}}, DataType::FLOAT};
      tensor_guid_t label_tensor =
          cg_builder.create_input(label_shape, CreateGrad::NO);
      GenericTensorAccessorW label_backing =
          allocator.allocate_tensor(label_shape);
      tensor_backing_map.insert({label_tensor, label_backing});
      std::optional<ModelTrainingInstance> model_training_instance =
          ModelTrainingInstance{
              LossAttrs{SparseCategoricalCrossEntropyLossAttrs{
                  /*replace_labels=*/false}},
              label_tensor,
              logit_tensor};
      LocalTrainingBacking local_backing(allocator,
                                         cg_builder.computation_graph,
                                         tensor_backing_map,
                                         runtime_arg_config,
                                         model_training_instance,
                                         optimizer_attrs);
      local_backing.execute_init();
      local_backing.execute_forward();
      local_backing.execute_backward();
    }

    SUBCASE("NonconfigurableLossAttrs") {
      tensor_guid_t label_tensor =
          cg_builder.create_input(input_shape, CreateGrad::NO);
      GenericTensorAccessorW label_backing =
          allocator.allocate_tensor(input_shape);
      tensor_backing_map.insert({label_tensor, label_backing});

      SUBCASE("LossFunction::CATEGORICAL_CROSSENTROPY") {
        std::optional<ModelTrainingInstance> model_training_instance =
            ModelTrainingInstance{LossAttrs{NonconfigurableLossAttrs{
                                      LossFunction::CATEGORICAL_CROSSENTROPY}},
                                  label_tensor,
                                  logit_tensor};
        LocalTrainingBacking local_backing(allocator,
                                           cg_builder.computation_graph,
                                           tensor_backing_map,
                                           runtime_arg_config,
                                           model_training_instance,
                                           optimizer_attrs);
        local_backing.execute_init();
        local_backing.execute_forward();
        local_backing.execute_backward();
      }

      SUBCASE("LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE") {
        std::optional<ModelTrainingInstance> model_training_instance =
            ModelTrainingInstance{
                LossAttrs{NonconfigurableLossAttrs{
                    LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE}},
                label_tensor,
                logit_tensor};
        LocalTrainingBacking local_backing(allocator,
                                           cg_builder.computation_graph,
                                           tensor_backing_map,
                                           runtime_arg_config,
                                           model_training_instance,
                                           optimizer_attrs);
        local_backing.execute_init();
        local_backing.execute_forward();
        local_backing.execute_backward();
      }

      SUBCASE("LossFunction::IDENTITY") {
        std::optional<ModelTrainingInstance> model_training_instance =
            ModelTrainingInstance{
                LossAttrs{NonconfigurableLossAttrs{LossFunction::IDENTITY}},
                label_tensor,
                logit_tensor};
        LocalTrainingBacking local_backing(allocator,
                                           cg_builder.computation_graph,
                                           tensor_backing_map,
                                           runtime_arg_config,
                                           model_training_instance,
                                           optimizer_attrs);
        local_backing.execute_init();
        local_backing.execute_forward();
        local_backing.execute_backward();
      }
    }
  }
}

} // namespace FlexFlow
