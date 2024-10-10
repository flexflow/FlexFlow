// #include "doctest/doctest.h"
// #include "kernels/local_cuda_allocator.h"
// #include "kernels/managed_per_device_ff_handle.h"
// #include "local-execution/local_cost_estimator.h"
// #include "op-attrs/ops/attention.h"
// #include "op-attrs/parallel_tensor_shape.h"
// #include "pcg/computation_graph_builder.h"
// #include "test_utils.h"

// using namespace ::FlexFlow;

// TEST_SUITE(FF_CUDA_TEST_SUITE) {
//   TEST_CASE("Local Cost Estimator") {
//     // local backing initialization
//     ManagedPerDeviceFFHandle managed_handle{};

//     RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
//         DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
//         EnableProfiling::YES,
//         ProfilingSettings{/*warmup_iters=*/0,
//                           /*measure_iters=*/1}};

//     LocalCostEstimator cost_estimator =
//     LocalCostEstimator{runtime_arg_config};

//     SUBCASE("Estimate cost -- Attention Op") {
//       int embed_dim = 32;
//       int num_heads = 10;
//       MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
//           /*embed_dim=*/embed_dim,
//           /*num_heads=*/num_heads,
//           /*kdim=*/embed_dim,
//           /*vdim=*/embed_dim,
//           /*dropout=*/0.0,
//           /*bias=*/true,
//           /*add_bias_kv=*/false,
//           /*add_zero_attn=*/false,
//       };

//       size_t batch_size = 40;
//       size_t seq_len = 48;
//       size_t feature_size = 36;

//       DataType dtype = DataType::FLOAT;
//       ParallelTensorShape inputs_shape = lift_to_parallel(TensorShape{
//           TensorDims{FFOrdered<size_t>{batch_size, seq_len, feature_size}},
//           DataType::FLOAT,
//       });

//       ParallelTensorShape weights_shape = throw_if_unexpected(
//           get_weights_shape(attrs, inputs_shape, inputs_shape,
//           inputs_shape));
//       ParallelTensorAttrs weight_attrs =
//           ParallelTensorAttrs{weights_shape,
//                               /*sync_type=*/std::nullopt,
//                               /*initializer=*/std::nullopt,
//                               CreateGrad::YES};

//       ParallelTensorShape output_shape = throw_if_unexpected(
//           get_output_shape(attrs, inputs_shape, inputs_shape, inputs_shape));
//       ParallelTensorAttrs output_attrs =
//           ParallelTensorAttrs{output_shape,
//                               /*sync_type=*/std::nullopt,
//                               /*initializer=*/std::nullopt,
//                               CreateGrad::YES};

//       CostDetails result = cost_estimator.estimate_cost(
//           PCGOperatorAttrs{attrs},
//           std::vector<ParallelTensorShape>{
//               inputs_shape, inputs_shape, inputs_shape},
//           std::vector<ParallelTensorAttrs>{weight_attrs},
//           std::vector<ParallelTensorAttrs>{output_attrs},
//           make_1d_machine_view(gpu_id_t{0}, gpu_id_t{1}));

//       CHECK(result.total_elapsed_time > 0);
//       CHECK(result.total_mem_usage > 0);
//     }
//   }
// }
