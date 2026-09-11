#include "local-execution/computation_graph_instance.h"
#include "kernels/compare_tensor_accessors.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/device_handle_t.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "kernels/tensor_accessor_reductions.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/device_type.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"
#include "test/utils/doctest/check_kv.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static bool did_loss_decrease(GenericTensorAccessorR const &first_epoch,
                              GenericTensorAccessorR const &last_epoch) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  return tensor_accessor_all(
      compare_tensor_accessors_le(last_epoch, first_epoch, cpu_allocator));
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("LocalBackend e2e Training") {
    Allocator allocator = create_local_cpu_memory_allocator();

    positive_int batch_size = 10_p;
    positive_int data_dim = 16_p;
    positive_int hidden_dim = 32_p;
    positive_int output_dim = 1_p;

    TensorShape output_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

    GenericTensorAccessorW label_tensor_backing =
        allocator.allocate_tensor(output_tensor_shape);

    // construct computation graph
    ComputationGraph computation_graph = make_empty_computation_graph();

    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

    TensorShape label_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};
    GenericTensorAccessorW label_tensor =
        allocator.allocate_tensor(label_tensor_shape);

    TensorShape weight_shape_1 = TensorShape{
        TensorDims{FFOrdered{hidden_dim, data_dim}}, DataType::FLOAT};
    TensorShape weight_shape_2 = TensorShape{
        TensorDims{FFOrdered{output_dim, hidden_dim}}, DataType::FLOAT};

    LayerAddedResult inputs_layer =
        add_input_layer(computation_graph, input_tensor_shape);
    tensor_guid_t t_input =
        require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult weights_layer_1 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                       weight_shape_1, InitializerAttrs{GlorotNormalAttrs{0}}}},
                   std::nullopt},
        {},
        {});
    tensor_guid_t t_weights_1 =
        require_only_key(weights_layer_1.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult weights_layer_2 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                       weight_shape_2, InitializerAttrs{GlorotNormalAttrs{0}}}},
                   std::nullopt},
        {},
        {});
    tensor_guid_t t_weights_2 =
        require_only_key(weights_layer_2.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult linear_operator_1 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{hidden_dim,
                                                       /*use_bias=*/false,
                                                       DataType::FLOAT,
                                                       Activation::RELU,
                                                       std::nullopt}},
                   std::nullopt},
        {
            {
                TensorSlotName::INPUT,
                t_input,
            },
        },
        {
            {
                TensorSlotName::WEIGHT,
                t_weights_1,
            },
        });
    tensor_guid_t t_linear_1 =
        require_only_key(linear_operator_1.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult linear_operator_2 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{output_dim,
                                                       /*use_bias=*/false,
                                                       DataType::FLOAT,
                                                       Activation::RELU,
                                                       std::nullopt}},
                   std::nullopt},
        {
            {
                TensorSlotName::INPUT,
                t_linear_1,
            },
        },
        {
            {
                TensorSlotName::WEIGHT,
                t_weights_2,
            },
        });
    tensor_guid_t t_linear_2 =
        require_only_key(linear_operator_2.outputs, TensorSlotName::OUTPUT);

    // instantiate computation graph
    LossAttrs loss_attrs = LossAttrs{
        NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
    OptimizerAttrs optimizer_attrs =
        OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                         /*momentum=*/0.9,
                                         /*nesterov=*/false,
                                         /*weight_decay=*/0.001}};
    device_handle_t ff_handle = cpu_make_device_handle_t();
    global_device_id_t global_device_id = global_device_id_t{
        /*coord=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*device_type=*/DeviceType::CPU,
    };

    std::map<DynamicValueAttrs, DynamicTensorAccessor> input_tensors;

    ComputationGraphInstance computation_graph_instance =
        create_computation_graph_instance(
            /*cg=*/computation_graph,
            /*optimizer=*/optimizer_attrs,
            /*loss=*/
            LossConfig{
                /*loss_attrs=*/loss_attrs,
                /*label_tensor=*/label_tensor,
                /*logit_tensor=*/t_linear_2,
            },
            /*input_tensors=*/input_tensors,
            /*allocator=*/allocator,
            /*device_handle=*/ff_handle,
            /*global_device_id=*/global_device_id);

    // begin training loop
    int num_epochs = 5;
    std::vector<GenericTensorAccessorR> loss_values;

    for (int i = 0; i < num_epochs; i++) {
      perform_all_passes_for_computation_graph_instance(
          /*instance=*/computation_graph_instance,
          /*profiling_settings=*/ProfilingSettings{0_n, 1_p},
          /*ff_handle=*/ff_handle,
          /*global_device_id=*/global_device_id);
      loss_values.push_back(copy_tensor_accessor_r(
          computation_graph_instance.get_loss_tensor_accessor().value(),
          allocator));
    }

    // Assert that each sample in the batch has a lower loss in last epoch than
    // the first epoch
    GenericTensorAccessorR first_epoch_loss = loss_values.at(0);
    GenericTensorAccessorR last_epoch_loss = loss_values.back();
    CHECK_MESSAGE(did_loss_decrease(first_epoch_loss, last_epoch_loss),
                  check_kv("first_epoch_loss",
                           format_accessor_r_contents(first_epoch_loss)),
                  check_kv("last_epoch_loss",
                           format_accessor_r_contents(last_epoch_loss)));
  }
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("LocalBackend e2e Training (CUDA)") {
    // initialize runtime
    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    Allocator allocator = create_local_cuda_memory_allocator();

    positive_int batch_size = 10_p;
    positive_int data_dim = 16_p;
    positive_int hidden_dim = 32_p;
    positive_int output_dim = 1_p;

    TensorShape output_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

    GenericTensorAccessorW label_tensor_backing =
        allocator.allocate_tensor(output_tensor_shape);

    // construct computation graph
    ComputationGraph computation_graph = make_empty_computation_graph();

    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

    TensorShape label_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};
    GenericTensorAccessorW label_tensor =
        allocator.allocate_tensor(label_tensor_shape);

    TensorShape weight_shape_1 = TensorShape{
        TensorDims{FFOrdered{data_dim, hidden_dim}}, DataType::FLOAT};
    TensorShape weight_shape_2 = TensorShape{
        TensorDims{FFOrdered{hidden_dim, output_dim}}, DataType::FLOAT};

    LayerAddedResult inputs_layer =
        add_input_layer(computation_graph, input_tensor_shape);
    tensor_guid_t t_input =
        require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult weights_layer_1 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                       weight_shape_1, InitializerAttrs{GlorotNormalAttrs{0}}}},
                   std::nullopt},
        {},
        {});
    tensor_guid_t t_weights_1 =
        require_only_key(weights_layer_1.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult weights_layer_2 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                       weight_shape_2, InitializerAttrs{GlorotNormalAttrs{0}}}},
                   std::nullopt},
        {},
        {});
    tensor_guid_t t_weights_2 =
        require_only_key(weights_layer_2.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult linear_operator_1 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{hidden_dim,
                                                       /*use_bias=*/false,
                                                       DataType::FLOAT,
                                                       Activation::RELU,
                                                       std::nullopt}},
                   std::nullopt},
        {
            {
                TensorSlotName::INPUT,
                t_input,
            },
        },
        {
            {
                TensorSlotName::WEIGHT,
                t_weights_1,
            },
        });
    tensor_guid_t t_linear_1 =
        require_only_key(linear_operator_1.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult linear_operator_2 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{output_dim,
                                                       /*use_bias=*/false,
                                                       DataType::FLOAT,
                                                       Activation::RELU,
                                                       std::nullopt}},
                   std::nullopt},
        {
            {
                TensorSlotName::INPUT,
                t_linear_1,
            },
        },
        {
            {
                TensorSlotName::WEIGHT,
                t_weights_2,
            },
        });
    tensor_guid_t t_linear_2 =
        require_only_key(linear_operator_2.outputs, TensorSlotName::OUTPUT);

    // instantiate computation graph
    LossAttrs loss_attrs = LossAttrs{
        NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
    OptimizerAttrs optimizer_attrs = OptimizerAttrs{
        SGDOptimizerAttrs{
            /*lr=*/0.001,
            /*momentum=*/0.9,
            /*nesterov=*/false,
            /*weight_decay=*/0.001,
        },
    };
    global_device_id_t device_idx = global_device_id_t{
        /*coord=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*device_type=*/DeviceType::GPU,
    };
    device_handle_t ff_handle =
        gpu_make_device_handle_t(managed_handle.raw_handle());

    std::map<DynamicValueAttrs, DynamicTensorAccessor> input_tensors;

    ComputationGraphInstance computation_graph_instance =
        create_computation_graph_instance(
            /*cg=*/computation_graph,
            /*optimizer=*/optimizer_attrs,
            /*loss=*/
            LossConfig{
                /*loss_attrs=*/loss_attrs,
                /*label_tensor=*/label_tensor,
                /*logit_tensor=*/t_linear_2,
            },
            /*input_tensors=*/input_tensors,
            /*allocator=*/allocator,
            /*device_handle=*/ff_handle,
            /*device_idx=*/device_idx);

    // begin training loop
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    int num_epochs = 5;
    std::vector<GenericTensorAccessorR> loss_values;

    for (int i = 0; i < num_epochs; i++) {
      perform_all_passes_for_computation_graph_instance(
          /*instance=*/computation_graph_instance,
          /*profiling_settings=*/ProfilingSettings{0_n, 1_p},
          /*ff_handle=*/ff_handle,
          /*device_idx=*/device_idx);
      loss_values.push_back(copy_tensor_accessor_r(
          computation_graph_instance.get_loss_tensor_accessor().value(),
          cpu_allocator));
    }

    // Assert that each sample in the batch has a lower loss in last epoch than
    // the first epoch
    GenericTensorAccessorR first_epoch_loss = loss_values.at(0);
    GenericTensorAccessorR last_epoch = loss_values.back();
    CHECK(did_loss_decrease(first_epoch_loss, last_epoch));
  }

  TEST_CASE("LossFunctions") {
    // initialize runtime
    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    Allocator allocator = create_local_cuda_memory_allocator();

    positive_int batch_size = 10_p;
    positive_int data_dim = 16_p;
    positive_int output_dim = 32_p;

    // construct computation graph
    ComputationGraph computation_graph = make_empty_computation_graph();

    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

    TensorShape weight_shape = TensorShape{
        TensorDims{FFOrdered{data_dim, output_dim}}, DataType::FLOAT};

    LayerAddedResult inputs_layer =
        add_input_layer(computation_graph, input_tensor_shape);
    tensor_guid_t inputs_tensor =
        require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult weights_layer = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                       weight_shape, InitializerAttrs{ZeroInitializerAttrs{}}}},
                   std::nullopt},
        {},
        {});
    tensor_guid_t weights_tensor =
        require_only_key(weights_layer.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult linear_operator = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{output_dim,
                                                       /*use_bias=*/false,
                                                       DataType::FLOAT,
                                                       Activation::RELU,
                                                       std::nullopt}},
                   std::nullopt},
        {
            {
                TensorSlotName::INPUT,
                inputs_tensor,
            },
        },
        {
            {
                TensorSlotName::WEIGHT,
                weights_tensor,
            },
        });
    tensor_guid_t logit_tensor =
        require_only_key(linear_operator.outputs, TensorSlotName::OUTPUT);

    OptimizerAttrs optimizer_attrs = OptimizerAttrs{
        SGDOptimizerAttrs{
            /*lr=*/0.0,
            /*momentum=*/0.0,
            /*nesterov=*/false,
            /*weight_decay=*/0.0,
        },
    };

    global_device_id_t device_idx = global_device_id_t{
        /*coord=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*device_type=*/DeviceType::GPU,
    };

    device_handle_t ff_handle =
        gpu_make_device_handle_t(managed_handle.raw_handle());

    std::map<DynamicValueAttrs, DynamicTensorAccessor> input_tensors;

    auto compute_loss = [&](LossAttrs const &loss_attrs,
                            GenericTensorAccessorR label_tensor) {
      ComputationGraphInstance computation_graph_instance =
          create_computation_graph_instance(
              /*cg=*/computation_graph,
              /*optimizer=*/optimizer_attrs,
              /*loss=*/
              LossConfig{
                  /*loss_attrs=*/loss_attrs,
                  /*label_tensor=*/label_tensor,
                  /*logit_tensor=*/logit_tensor,
              },
              /*input_tensors=*/input_tensors,
              /*allocator=*/allocator,
              /*device_handle=*/ff_handle,
              /*device_idx=*/device_idx);

      perform_all_passes_for_computation_graph_instance(
          /*instance=*/computation_graph_instance,
          /*profiling_settings=*/ProfilingSettings{0_n, 1_p},
          /*ff_handle=*/ff_handle,
          /*device_idx=*/device_idx);
      assert_unwrap(computation_graph_instance.get_loss_tensor_accessor());
    };

    SUBCASE("SparseCategoricalCrossEntropyLossAttrs") {
      TensorShape label_tensor_shape =
          TensorShape{TensorDims{FFOrdered{batch_size, 1_p}}, DataType::FLOAT};
      GenericTensorAccessorW label_tensor =
          allocator.allocate_tensor(label_tensor_shape);

      LossAttrs loss_attrs = LossAttrs{
          SparseCategoricalCrossEntropyLossAttrs{/*replace_labels=*/false}};

      compute_loss(loss_attrs, label_tensor);
    }

    SUBCASE("NonconfigurableLossAttrs") {
      TensorShape label_tensor_shape = TensorShape{
          TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};
      GenericTensorAccessorW label_tensor =
          allocator.allocate_tensor(label_tensor_shape);

      SUBCASE("LossFunction::CATEGORICAL_CROSSENTROPY") {
        LossAttrs loss_attrs = LossAttrs{
            NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};

        compute_loss(loss_attrs, label_tensor);
      }

      SUBCASE("LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE") {
        LossAttrs loss_attrs = LossAttrs{NonconfigurableLossAttrs{
            LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE}};

        compute_loss(loss_attrs, label_tensor);
      }

      SUBCASE("LossFunction::IDENTITY") {
        LossAttrs loss_attrs =
            LossAttrs{NonconfigurableLossAttrs{LossFunction::IDENTITY}};

        compute_loss(loss_attrs, label_tensor);
      }
    }
  }
}
