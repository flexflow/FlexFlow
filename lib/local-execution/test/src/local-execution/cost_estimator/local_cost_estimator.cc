#include "local-execution/cost_estimator/local_cost_estimator.h"
#include "compiler/machine_mapping/machine_view.h"
#include "kernels/device_handle_t.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/cast.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/computation_graph_builder.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("LocalCostEstimator") {
    Allocator allocator = create_local_cpu_memory_allocator();
    device_handle_t ff_handle = cpu_make_device_handle_t();
    global_device_id_t device_idx = global_device_id_t{
        /*coord=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*device_type=*/DeviceType::CPU,
    };

    OptimizerAttrs optimizer_attrs = OptimizerAttrs{
        SGDOptimizerAttrs{
            /*lr=*/0.1,
            /*momentum=*/0.1,
            /*nesterov=*/false,
            /*weight_decay=*/0.1,
        },
    };

    MachineInterconnectSpecification interconnect_specification{
        /*inter_node_bandwidth=*/bytes_per_second_t{10000000},
        /*intra_node_bandwidth=*/bytes_per_second_t{10000000000},
    };

    CostEstimator cost_estimator = get_local_cost_estimator(
        /*interconnect_specification=*/interconnect_specification,
        /*allocator=*/allocator,
        /*profiling_settings=*/
        ProfilingSettings{/*warmup_iters=*/0_n,
                          /*measure_iters=*/1_p},
        /*device_handle=*/ff_handle,
        /*device_idx=*/device_idx);

    SUBCASE("estimate operator cost") {
      CastAttrs attrs = CastAttrs{
          /*dtype=*/DataType::INT32,
      };

      ParallelTensorShape input_shape = lift_to_parallel(TensorShape{
          TensorDims{FFOrdered<positive_int>{40_p, 48_p, 36_p}},
          DataType::FLOAT,
      });

      ParallelTensorShape output_shape =
          throw_if_unexpected(get_output_shape(attrs, input_shape));

      OpCostEstimateKey op_cost_estimate_key = OpCostEstimateKey{
          /*op_attrs=*/PCGOperatorAttrs{attrs},
          /*input_shapes=*/{{TensorSlotName::INPUT, input_shape}},
          /*weight_shapes=*/{},
          /*output_shapes=*/{{TensorSlotName::OUTPUT, output_shape}},
          /*optimizer_attrs=*/optimizer_attrs,
          /*machine_view=*/
          make_1d_machine_view(MachineSpaceCoordinate{0_n, 0_n},
                               MachineSpecificationDimension::INTRA_NODE,
                               stride_t{1_p}),
      };

      OpCostMetrics result = cost_estimator.estimate_cost(op_cost_estimate_key);

      CHECK(result.forward_runtime > 0_ms);
      CHECK(result.backward_runtime > 0_ms);
      CHECK(result.memory_usage > 0_bytes);
    }
  }
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("LocalCostEstimator (CUDA)") {
    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    Allocator allocator = create_local_cuda_memory_allocator();

    global_device_id_t device_idx = global_device_id_t{
        /*coord=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*device_type=*/DeviceType::GPU,
    };

    device_handle_t ff_handle =
        gpu_make_device_handle_t(managed_handle.raw_handle());

    OptimizerAttrs optimizer_attrs = OptimizerAttrs{
        SGDOptimizerAttrs{
            /*lr=*/0.1,
            /*momentum=*/0.1,
            /*nesterov=*/false,
            /*weight_decay=*/0.1,
        },
    };

    MachineInterconnectSpecification interconnect_specification{
        /*inter_node_bandwidth=*/bytes_per_second_t{10000000},
        /*intra_node_bandwidth=*/bytes_per_second_t{10000000000},
    };

    CostEstimator cost_estimator = get_local_cost_estimator(
        /*interconnect_specification=*/interconnect_specification,
        /*allocator=*/allocator,
        /*profiling_settings=*/
        ProfilingSettings{/*warmup_iters=*/0_n,
                          /*measure_iters=*/1_p},
        /*device_handle=*/ff_handle,
        /*device_idx=*/device_idx);

    SUBCASE("estimate operator cost") {
      positive_int embed_dim = 32_p;
      positive_int num_heads = 10_p;
      MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
          /*embed_dim=*/embed_dim,
          /*num_heads=*/num_heads,
          /*kdim=*/embed_dim,
          /*vdim=*/embed_dim,
          /*dropout=*/0.0,
          /*bias=*/false,
          /*add_bias_kv=*/false,
          /*add_zero_attn=*/false,
      };

      positive_int batch_size = 40_p;
      positive_int seq_len = 48_p;
      positive_int feature_size = 36_p;

      DataType dtype = DataType::FLOAT;
      ParallelTensorShape inputs_shape = lift_to_parallel(TensorShape{
          TensorDims{
              FFOrdered<positive_int>{batch_size, seq_len, feature_size}},
          DataType::FLOAT,
      });

      ParallelTensorShape weights_shape = throw_if_unexpected(
          get_weights_shape(attrs, inputs_shape, inputs_shape, inputs_shape));

      ParallelTensorShape output_shape = throw_if_unexpected(
          get_output_shape(attrs, inputs_shape, inputs_shape, inputs_shape));

      OpCostEstimateKey op_cost_estimate_key = OpCostEstimateKey{
          /*op_attrs=*/PCGOperatorAttrs{attrs},
          /*input_shapes=*/
          {{TensorSlotName::QUERY, inputs_shape},
           {TensorSlotName::KEY, inputs_shape},
           {TensorSlotName::VALUE, inputs_shape}},
          /*weight_shapes=*/{{TensorSlotName::WEIGHT, weights_shape}},
          /*output_shapes=*/{{TensorSlotName::OUTPUT, output_shape}},
          /*optimizer_attrs=*/optimizer_attrs,
          /*machine_view=*/
          make_1d_machine_view(MachineSpaceCoordinate{0_n, 0_n},
                               MachineSpecificationDimension::INTRA_NODE,
                               stride_t{1_p}),
      };

      OpCostMetrics result = cost_estimator.estimate_cost(op_cost_estimate_key);

      CHECK(result.forward_runtime > 0_ms);
      CHECK(result.backward_runtime > 0_ms);
      CHECK(result.memory_usage > 0_bytes);
    }
  }
}
