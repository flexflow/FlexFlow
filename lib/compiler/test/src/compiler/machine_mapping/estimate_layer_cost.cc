#include "compiler/machine_mapping/estimate_layer_cost.h"
#include "./cost_estimator_for_test.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("estimate_layer_cost") {
    ParallelTensorShape input_shape = ParallelTensorShape{
      ParallelTensorDims{
        FFOrdered<ShardParallelDim>{
          ShardParallelDim{8, 2},
          ShardParallelDim{10, 1},
        },
        ReplicaParallelDimSet{
          SumDegree{1},
          DiscardCopyDegree{1},
        },
      },
      DataType::FLOAT,
    };

    LinearAttrs linear_attrs = LinearAttrs{
      /*out_channels=*/12,
      /*use_bias=*/true,
      /*data_type=*/DataType::FLOAT,
      /*activation=*/std::nullopt,
      /*regularizer=*/std::nullopt,
    };

    ParallelTensorShape projection_shape = throw_if_unexpected(get_projection_shape(linear_attrs, input_shape));
    ParallelTensorShape bias_shape = throw_if_unexpected(get_bias_shape(linear_attrs, input_shape));
    ParallelTensorShape output_shape = throw_if_unexpected(get_output_shape(linear_attrs, input_shape));

    auto make_tensor_attrs = [](ParallelTensorShape const &shape) {
      return ParallelTensorAttrs{
        /*shape=*/shape, 
        /*sync_type=*/std::nullopt,
        /*initializer=*/std::nullopt,
        /*create_grad=*/CreateGrad::YES,
      };
    };

    auto make_layer_attrs = [](PCGOperatorAttrs const &op_attrs) {
      return ParallelLayerAttrs{
        /*op_attrs=*/op_attrs,
        /*name=*/std::nullopt,
      };
    };

    ParallelComputationGraph pcg = empty_parallel_computation_graph();
    ParallelLayerAddedResult input = add_parallel_layer(pcg,
                                                        /*layer_attrs=*/make_layer_attrs(PCGOperatorAttrs{InputAttrs{}}),
                                                        /*inputs=*/{},
                                                        /*output_labels=*/{make_tensor_attrs(input_shape)});
    parallel_tensor_guid_t input_tensor = get_only(input.outputs);

    ParallelLayerAddedResult projection = add_parallel_layer(pcg, 
                                                             /*layer_attrs=*/make_layer_attrs(
                                                                PCGOperatorAttrs{
                                                                  WeightAttrs{
                                                                    /*tensor_shape=*/get_reduced_shape(projection_shape),
                                                                  },
                                                                }),
                                                             /*inputs=*/{},
                                                             /*output_labels=*/{make_tensor_attrs(projection_shape)});
    parallel_tensor_guid_t projection_tensor = get_only(projection.outputs);

    ParallelLayerAddedResult bias = add_parallel_layer(pcg,
                                                       /*layer_attrs=*/make_layer_attrs(
                                                          PCGOperatorAttrs{
                                                            WeightAttrs{
                                                              /*tensor_shape=*/get_reduced_shape(bias_shape),
                                                            },
                                                          }),
                                                       /*inputs=*/{},
                                                       /*output_labels=*/{make_tensor_attrs(bias_shape)});
    parallel_tensor_guid_t bias_tensor = get_only(bias.outputs);

    ParallelLayerAddedResult linear = add_parallel_layer(pcg,
                                                         /*layer_attrs=*/make_layer_attrs(PCGOperatorAttrs{linear_attrs}),
                                                         /*inputs=*/{
                                                           get_only(input.outputs), 
                                                           get_only(projection.outputs),
                                                           get_only(bias.outputs),
                                                         },
                                                         /*output_labels=*/{make_tensor_attrs(output_shape)});
    parallel_tensor_guid_t linear_output = get_only(linear.outputs);

    MachineView machine_view = make_1d_machine_view(gpu_id_t{0}, gpu_id_t{1});
                                                                        

    CostEstimator cost_estimator = make_fake_cost_estimator(
      {
        {
          OpCostEstimateKey{
            /*op_attrs=*/PCGOperatorAttrs{linear_attrs},
            /*input_shapes=*/{input_shape},
            /*weight_shapes=*/{projection_shape, bias_shape},
            /*output_shapes=*/{output_shape},
            /*machine_view=*/machine_view,         
          },
          2.0,
        },
      },
      {}
    );

    SUBCASE("returns just the layer cost if the layer exists") {
      float result = estimate_layer_cost(pcg,
                                         cost_estimator,
                                         linear.parallel_layer,
                                         machine_view);
      float correct = 2.0;

      CHECK(result == correct);
    }
  }
}
