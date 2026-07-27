#include "internal/realm_test_utils.h"
#include "kernels/allocation.h"
#include "kernels/compare_tensor_accessors.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/tensor_accessor_reductions.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/device_type.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "realm-execution/distributed_ff_handle.h"
#include "realm-execution/dynamic_tensor_accessor_from_instance.h"
#include "realm-execution/pcg_instance.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/realm_manager.h"
#include "task-spec/permissions.h"
#include "test/utils/doctest/check_kv.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

namespace test {

using namespace ::FlexFlow;
namespace Realm = ::FlexFlow::Realm;

template <typename T>
static ParallelLayerAttrs make_layer_attrs(T const &op_attrs) {
  return ParallelLayerAttrs{
      /*op_attrs=*/PCGOperatorAttrs{op_attrs},
      /*name=*/std::nullopt,
  };
};

static bool did_loss_decrease(GenericTensorAccessorR const &first_epoch,
                              GenericTensorAccessorR const &last_epoch,
                              Allocator &allocator) {
  return tensor_accessor_all(
      compare_tensor_accessors_le(last_epoch, first_epoch, allocator));
}

struct E2ETrainingConfig {
  MappedParallelComputationGraph mapped_pcg;
  LossAttrs loss_attrs;
  MappedOperatorTaskGroup loss_mapping;
  OptimizerAttrs optimizer_attrs;
  parallel_tensor_guid_t logit_tensor;
  TensorShape input_shape;
  TensorShape logit_shape;
  TensorShape label_shape;
  TensorShape loss_shape;
};

static E2ETrainingConfig create_e2e_test_case() {
  positive_int batch_size = 10_p;
  positive_int data_dim = 16_p;
  positive_int hidden_dim = 32_p;
  positive_int output_dim = 1_p;

  TensorShape input_tensor_shape =
      TensorShape{TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

  TensorShape label_tensor_shape = TensorShape{
      TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

  TensorShape loss_tensor_shape = TensorShape{
      TensorDims{FFOrdered{output_dim, hidden_dim}}, DataType::FLOAT};

  TensorShape weight_shape_1 =
      TensorShape{TensorDims{FFOrdered{hidden_dim, data_dim}}, DataType::FLOAT};

  TensorShape weight_shape_2 = TensorShape{
      TensorDims{FFOrdered{output_dim, hidden_dim}}, DataType::FLOAT};

  ParallelComputationGraph pcg = empty_parallel_computation_graph();

  ParallelLayerAddedResult inputs_layer =
      pcg_add_input_layer(pcg, input_tensor_shape);
  parallel_tensor_guid_t t_input =
      require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

  ParallelLayerAddedResult weights_layer_1 = add_parallel_layer(
      pcg,
      ParallelLayerAttrs{
          PCGOperatorAttrs{WeightAttrs{weight_shape_1,
                                       InitializerAttrs{GlorotNormalAttrs{0}}}},
          std::nullopt},
      {},
      {});
  parallel_tensor_guid_t t_weights_1 =
      require_only_key(weights_layer_1.outputs, TensorSlotName::OUTPUT);

  ParallelLayerAddedResult weights_layer_2 = add_parallel_layer(
      pcg,
      ParallelLayerAttrs{
          PCGOperatorAttrs{WeightAttrs{weight_shape_2,
                                       InitializerAttrs{GlorotNormalAttrs{0}}}},
          std::nullopt},
      {},
      {});
  parallel_tensor_guid_t t_weights_2 =
      require_only_key(weights_layer_2.outputs, TensorSlotName::OUTPUT);

  ParallelLayerAddedResult linear_operator_1 = add_parallel_layer(
      pcg,
      ParallelLayerAttrs{PCGOperatorAttrs{LinearAttrs{hidden_dim,
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
  parallel_tensor_guid_t t_linear_1 =
      require_only_key(linear_operator_1.outputs, TensorSlotName::OUTPUT);

  ParallelLayerAddedResult linear_operator_2 = add_parallel_layer(
      pcg,
      ParallelLayerAttrs{PCGOperatorAttrs{LinearAttrs{output_dim,
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
  parallel_tensor_guid_t t_linear_2 =
      require_only_key(linear_operator_2.outputs, TensorSlotName::OUTPUT);

  MachineSpaceCoordinate cpu0{0_n, 0_n};
  MachineSpaceCoordinate cpu1{0_n, 1_n};
  ParallelTensorSpaceCoordinate tensor_coord0{0_n, 0_n, FFOrdered{0_n}};

  std::map<parallel_layer_guid_t, MappedOperatorTaskGroup> mapping = {
      {inputs_layer.parallel_layer,
       MappedOperatorTaskGroup{
           {{cpu0,
             OperatorAtomicTaskShardBinding{
                 {{TensorSlotName::OUTPUT, tensor_coord0}}}}}}},
      {weights_layer_1.parallel_layer,
       MappedOperatorTaskGroup{
           {{cpu0,
             OperatorAtomicTaskShardBinding{
                 {{TensorSlotName::OUTPUT, tensor_coord0}}}}}}},
      {weights_layer_2.parallel_layer,
       MappedOperatorTaskGroup{
           {{cpu1,
             OperatorAtomicTaskShardBinding{
                 {{TensorSlotName::OUTPUT, tensor_coord0}}}}}}},
      {linear_operator_1.parallel_layer,
       MappedOperatorTaskGroup{{{cpu0,
                                 OperatorAtomicTaskShardBinding{{
                                     {TensorSlotName::INPUT, tensor_coord0},
                                     {TensorSlotName::WEIGHT, tensor_coord0},
                                     {TensorSlotName::OUTPUT, tensor_coord0},
                                 }}}}}},
      {linear_operator_2.parallel_layer,
       MappedOperatorTaskGroup{{{cpu1,
                                 OperatorAtomicTaskShardBinding{{
                                     {TensorSlotName::INPUT, tensor_coord0},
                                     {TensorSlotName::WEIGHT, tensor_coord0},
                                     {TensorSlotName::OUTPUT, tensor_coord0},
                                 }}}}}},
  };

  TensorShape output_tensor_shape = TensorShape{
      TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

  MappedParallelComputationGraph mpcg =
      mapped_pcg_from_pcg_and_mapped_op_task_groups(pcg, mapping);

  MappedOperatorTaskGroup loss_mapping{
      {{cpu0,
        OperatorAtomicTaskShardBinding{{
            {TensorSlotName::INPUT, tensor_coord0},
            {TensorSlotName::LOGIT, tensor_coord0},
        }}}}};

  LossAttrs loss_attrs = LossAttrs{
      NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
  OptimizerAttrs optimizer_attrs =
      OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                       /*momentum=*/0.9,
                                       /*nesterov=*/false,
                                       /*weight_decay=*/0.001}};

  return E2ETrainingConfig{
      /*mapped_pcg=*/mpcg,
      /*loss_attrs=*/loss_attrs,
      /*loss_mapping=*/loss_mapping,
      /*optimizer_attrs=*/optimizer_attrs,
      /*logit_tensor=*/t_linear_2,
      /*input_shape=*/input_tensor_shape,
      /*logit_shape=*/output_tensor_shape,
      /*label_shape=*/label_tensor_shape,
      /*loss_shape=*/loss_tensor_shape,
  };
}

MappedParallelComputationGraph
    make_test_replicate_mpcg_for_device_type(DeviceType device_type) {
  positive_int batch_size = 10_p;
  positive_int data_dim = 16_p;
  positive_int hidden_dim = 32_p;
  positive_int output_dim = 1_p;

  TensorShape output_tensor_shape = TensorShape{
      TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

  TensorShape label_tensor_shape = TensorShape{
      TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

  ParallelComputationGraph pcg = empty_parallel_computation_graph();

  TensorShape input_tensor_shape =
      TensorShape{TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

  ParallelLayerAddedResult inputs_layer =
      pcg_add_input_layer(pcg, input_tensor_shape);
  parallel_tensor_guid_t t_input =
      require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

  ParallelLayerAddedResult inputs_layer_2 =
      pcg_add_input_layer(pcg, input_tensor_shape);
  parallel_tensor_guid_t t_input_2 =
      require_only_key(inputs_layer_2.outputs, TensorSlotName::OUTPUT);

  ElementBinaryAttrs add_attrs = ElementBinaryAttrs{
      OperatorType::EW_ADD,
      DataType::FLOAT,
      false,
      false,
  };

  ParallelLayerAddedResult add_operator_1 =
      add_parallel_layer(pcg,
                         make_layer_attrs(add_attrs),
                         {
                             {
                                 TensorSlotName::LHS_INPUT,
                                 t_input,
                             },
                             {
                                 TensorSlotName::RHS_INPUT,
                                 t_input_2,
                             },
                         },
                         /*weights=*/{});

  parallel_tensor_guid_t t_add_1 =
      require_only_key(add_operator_1.outputs, TensorSlotName::OUTPUT);

  positive_int replicate_degree = 2_p;
  ReplicateAttrs repl_attrs = ReplicateAttrs{replicate_degree};
  ParallelLayerAddedResult repl_operator_1 =
      add_parallel_layer(pcg,
                         make_layer_attrs(repl_attrs),
                         {
                             {
                                 TensorSlotName::INPUT,
                                 t_add_1,
                             },
                         },
                         /*weight=*/{});

  parallel_tensor_guid_t t_repl_1 =
      require_only_key(repl_operator_1.outputs, TensorSlotName::OUTPUT);

  ParallelLayerAddedResult relu_operator_1 =
      add_parallel_layer(pcg,
                         make_layer_attrs(make_relu_attrs()),
                         /*inputs=*/
                         {
                             {
                                 TensorSlotName::INPUT,
                                 t_repl_1,
                             },
                         },
                         /*weights=*/{});

  parallel_tensor_guid_t t_relu_1 =
      require_only_key(relu_operator_1.outputs, TensorSlotName::OUTPUT);

  MachineSpaceCoordinate mc0{0_n, 0_n};
  MachineSpaceCoordinate mc1{0_n, 1_n};

  ParallelTensorSpaceCoordinate tensor_coord0{
      /*sum_component=*/0_n,
      /*discard_copy_component=*/0_n,
      /*shard_component=*/FFOrdered{0_n}};
  ParallelTensorSpaceCoordinate tensor_coord1{
      /*sum_component=*/0_n,
      /*discard_copy_component=*/1_n,
      /*shard_component=*/FFOrdered{0_n}};

  MappedParallelComputationGraph mpcg =
      mapped_pcg_from_pcg_and_mapped_op_task_groups(
          /*pcg=*/pcg,
          /*mapped_op_task_groups=*/{
              {
                  inputs_layer.parallel_layer,
                  MappedOperatorTaskGroup{
                      {
                          {
                              mc0,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::OUTPUT, tensor_coord0},
                              }},
                          },
                      },
                  },
              },
              {
                  inputs_layer_2.parallel_layer,
                  MappedOperatorTaskGroup{
                      {
                          {
                              mc0,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::OUTPUT, tensor_coord0},
                              }},
                          },
                      },
                  },
              },
              {
                  add_operator_1.parallel_layer,
                  MappedOperatorTaskGroup{
                      {
                          {
                              mc0,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::LHS_INPUT, tensor_coord0},
                                  {TensorSlotName::RHS_INPUT, tensor_coord0},
                                  {TensorSlotName::OUTPUT, tensor_coord0},
                              }},
                          },
                      },
                  },
              },
              {
                  repl_operator_1.parallel_layer,
                  MappedOperatorTaskGroup{
                      {
                          {
                              mc0,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::INPUT, tensor_coord0},
                                  {TensorSlotName::OUTPUT, tensor_coord0},
                              }},
                          },
                          {
                              mc1,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::INPUT, tensor_coord0},
                                  {TensorSlotName::OUTPUT, tensor_coord1},
                              }},
                          },
                      },
                  },
              },
              {
                  relu_operator_1.parallel_layer,
                  MappedOperatorTaskGroup{
                      {
                          {
                              mc0,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::INPUT, tensor_coord0},
                                  {TensorSlotName::OUTPUT, tensor_coord0},
                              }},
                          },
                          {
                              mc1,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::INPUT, tensor_coord1},
                                  {TensorSlotName::OUTPUT, tensor_coord1},
                              }},
                          },
                      },
                  },
              },
          });

  return mpcg;
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("RealmBackend e2e Training (CPU Model Parallelism)") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/2_p, /*num_gpus=*/0_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager = RealmManager{&fake_argc, &fake_argv};

    (void)manager.start_controller([](RealmContext &ctx) {
      E2ETrainingConfig cfg = create_e2e_test_case();

      Allocator allocator = ctx.get_current_device_allocator();

      GenericTensorAccessorW output_tensor =
          allocator.allocate_tensor(cfg.logit_shape);
      GenericTensorAccessorW label_tensor =
          allocator.allocate_tensor(cfg.label_shape);

      std::map<DynamicValueAttrs, DynamicTensorAccessor> input_tensors;

      DistributedFfHandle device_handle =
          create_distributed_ff_handle(ctx,
                                       /*workSpaceSize=*/1024 * 1024,
                                       /*allowTensorOpMathConversion=*/true);

      PCGInstance pcg_instance = create_pcg_instance(
          /*ctx=*/ctx,
          /*mpcg=*/cfg.mapped_pcg,
          /*optimizer=*/cfg.optimizer_attrs,
          /*loss=*/
          ParallelLossConfig{
              /*loss_attrs=*/cfg.loss_attrs,
              /*label_tensor=*/label_tensor,
              /*logit_tensor=*/cfg.logit_tensor,
              /*loss_mapping=*/cfg.loss_mapping,
          },
          /*input_tensors=*/input_tensors,
          /*profiling_settings=*/ProfilingSettings{0, 0},
          /*device_handle=*/device_handle,
          /*device_type=*/DeviceType::CPU);

      // begin training loop
      int num_epochs = 5;
      std::vector<GenericTensorAccessorR> loss_values;

      for (int i = 0; i < num_epochs; i++) {
        perform_all_passes_for_pcg_instance(
            /*instance=*/pcg_instance,
            /*profiling_settings=*/ProfilingSettings{0, 0},
            /*device_handle=*/device_handle);
        loss_values.push_back(copy_tensor_accessor_r(
            dynamic_tensor_accessor_from_instance(
                pcg_instance.get_loss_tensor_instance().value(),
                Realm::Event::NO_EVENT,
                lift_to_parallel(cfg.loss_shape),
                Permissions::RO,
                ctx.get_current_processor())
                .require_read(),
            allocator));
      }

      // Assert that each sample in the batch has a lower loss in last epoch
      // than the first epoch
      GenericTensorAccessorR first_epoch_loss = loss_values.at(0);
      GenericTensorAccessorR last_epoch_loss = loss_values.back();
      CHECK_MESSAGE(
          did_loss_decrease(first_epoch_loss, last_epoch_loss, allocator),
          check_kv("first_epoch_loss",
                   format_accessor_r_contents(first_epoch_loss)),
          check_kv("last_epoch_loss",
                   format_accessor_r_contents(last_epoch_loss)));
    });
  }

  TEST_CASE("RealmBackend e2e Training Replicate Op (CPU Model Parallelism)") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/2_p, /*num_gpus=*/0_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager = RealmManager{&fake_argc, &fake_argv};
    ControllerTaskResult result =
        manager.start_controller([](RealmContext &ctx) {
          Allocator allocator = ctx.get_current_device_allocator();

          MappedParallelComputationGraph mpcg =
              make_test_replicate_mpcg_for_device_type(DeviceType::CPU);

          std::map<DynamicValueAttrs, DynamicTensorAccessor> input_tensors;

          OptimizerAttrs optimizer_attrs = OptimizerAttrs{
              SGDOptimizerAttrs{
                  /*lr=*/0.001,
                  /*momentum=*/0.9,
                  /*nesterov=*/false,
                  /*weight_decay=*/0.001,
              },
          };

          DistributedFfHandle device_handle = create_distributed_ff_handle(
              ctx,
              /*workSpaceSize=*/1024 * 1024,
              /*allowTensorOpMathConversion=*/true);

          PCGInstance pcg_instance = create_pcg_instance(
              /*ctx=*/ctx,
              /*mpcg=*/mpcg,
              /*optimizer=*/optimizer_attrs,
              /*loss=*/std::nullopt,
              /*input_tensors=*/input_tensors,
              /*profiling_settings=*/ProfilingSettings{0, 0},
              /*device_handle=*/device_handle,
              /*device_type=*/DeviceType::CPU);

          // begin training loop
          int num_epochs = 1;
          for (int i = 0; i < num_epochs; i++) {
            perform_all_passes_for_pcg_instance(
                /*instance=*/pcg_instance,
                /*profiling_settings=*/ProfilingSettings{0, 0},
                /*device_handle=*/device_handle);
          }
        });
    result.wait();
  }
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("RealmBackend e2e Training (GPU Model Parallelism)") {
    E2ETrainingConfig cfg = create_e2e_test_case();

    //! [realm-execution example]
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/1_p, /*num_gpus=*/2_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager(&fake_argc, &fake_argv);

    ControllerTaskResult result =
        manager.start_controller([&](RealmContext &ctx) {
          Allocator allocator = ctx.get_current_device_allocator();

          GenericTensorAccessorW logit_tensor =
              allocator.allocate_tensor(cfg.logit_shape);
          GenericTensorAccessorW label_tensor =
              allocator.allocate_tensor(cfg.label_shape);

          std::map<DynamicValueAttrs, DynamicTensorAccessor> input_tensors;

          DistributedFfHandle device_handle = create_distributed_ff_handle(
              ctx,
              /*workSpaceSize=*/1024 * 1024,
              /*allowTensorOpMathConversion=*/true);

          PCGInstance pcg_instance = create_pcg_instance(
              /*ctx=*/ctx,
              /*mpcg=*/cfg.mapped_pcg,
              /*optimizer=*/cfg.optimizer_attrs,
              /*loss=*/
              ParallelLossConfig{
                  /*loss_attrs=*/cfg.loss_attrs,
                  /*label_tensor=*/label_tensor,
                  /*logit_tensor=*/cfg.logit_tensor,
                  /*loss_mapping=*/cfg.loss_mapping,
              },
              /*input_tensors=*/input_tensors,
              /*profiling_settings=*/ProfilingSettings{0, 0},
              /*device_handle=*/device_handle,
              /*device_type=*/DeviceType::GPU);

          // begin training loop
          int num_epochs = 5;
          std::vector<GenericTensorAccessorR> loss_values;

          for (int i = 0; i < num_epochs; i++) {
            perform_all_passes_for_pcg_instance(
                /*instance=*/pcg_instance,
                /*profiling_settings=*/ProfilingSettings{0, 0},
                /*device_handle=*/device_handle);

            loss_values.push_back(copy_tensor_accessor_r(
                dynamic_tensor_accessor_from_instance(
                    pcg_instance.get_loss_tensor_instance().value(),
                    Realm::Event::NO_EVENT,
                    lift_to_parallel(cfg.loss_shape),
                    Permissions::RO,
                    ctx.get_current_processor())
                    .require_read(),
                allocator));
          }

          // Assert that each sample in the batch has a lower loss in last epoch
          // than the first epoch
          GenericTensorAccessorR first_epoch_loss = loss_values.at(0);
          GenericTensorAccessorR last_epoch_loss = loss_values.back();
          CHECK_MESSAGE(
              did_loss_decrease(first_epoch_loss, last_epoch_loss, allocator),
              check_kv("first_epoch_loss",
                       format_accessor_r_contents(first_epoch_loss)),
              check_kv("last_epoch_loss",
                       format_accessor_r_contents(last_epoch_loss)));
        });

    result.wait();
    //! [realm-execution example]
  }

  TEST_CASE("RealmBackend e2e Training Replicate Op (GPU Model Parallelism)") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/1_p, /*num_gpus=*/2_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager = RealmManager{&fake_argc, &fake_argv};

    ControllerTaskResult result =
        manager.start_controller([](RealmContext &ctx) {
          Allocator allocator = ctx.get_current_device_allocator();

          MappedParallelComputationGraph mpcg =
              make_test_replicate_mpcg_for_device_type(DeviceType::GPU);

          OptimizerAttrs optimizer_attrs = OptimizerAttrs{
              SGDOptimizerAttrs{
                  /*lr=*/0.001,
                  /*momentum=*/0.9,
                  /*nesterov=*/false,
                  /*weight_decay=*/0.001,
              },
          };

          std::map<DynamicValueAttrs, DynamicTensorAccessor> input_tensors;

          DistributedFfHandle device_handle = create_distributed_ff_handle(
              ctx,
              /*workSpaceSize=*/1024 * 1024,
              /*allowTensorOpMathConversion=*/true);

          PCGInstance pcg_instance = create_pcg_instance(
              /*ctx=*/ctx,
              /*mpcg=*/mpcg,
              /*optimizer=*/optimizer_attrs,
              /*loss=*/std::nullopt,
              /*input_tensors=*/input_tensors,
              /*profiling_settings=*/ProfilingSettings{0, 0},
              /*device_handle=*/device_handle,
              /*device_type=*/DeviceType::GPU);

          // begin training loop
          int num_epochs = 1;
          for (int i = 0; i < num_epochs; i++) {
            perform_all_passes_for_pcg_instance(
                /*instance=*/pcg_instance,
                /*profiling_settings=*/ProfilingSettings{0, 0},
                /*device_handle=*/device_handle);
          }
        });
    result.wait();
  }
}

} // namespace test
