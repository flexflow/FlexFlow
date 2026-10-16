#include "pcg/pcg_from_computation_graph.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/linear.h"
#include "pcg/computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("pcg_from_computation_graph") {
    InitializerAttrs zero_init = InitializerAttrs{ZeroInitializerAttrs{}};

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                12_p,
            },
        },
        DataType::FLOAT,
    };

    LinearAttrs linear_attrs = LinearAttrs{
        /*out_channels=*/8_p,
        /*use_bias=*/true,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/Activation::RELU,
        /*regularizer=*/std::nullopt,
    };

    WeightAttrs projection_weight_attrs = WeightAttrs{
        /*tensor_shape=*/
        get_projection_shape(linear_attrs, input_shape),
        /*initializer=*/zero_init,
    };

    WeightAttrs bias_weight_attrs = WeightAttrs{
        /*tensor_shape=*/
        get_bias_shape(linear_attrs, input_shape),
        /*initializer=*/zero_init,
    };

    ComputationGraph cg = [&] {
      auto make_layer_attrs = [](auto const &op_attrs) {
        return LayerAttrs{
            /*op_attrs=*/ComputationGraphOpAttrs{op_attrs},
            /*name=*/std::nullopt,
        };
      };

      ComputationGraph cg = make_empty_computation_graph();

      LayerAddedResult input_added = add_input_layer(cg, input_shape);
      tensor_guid_t t_input =
          require_only_key(input_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult projection_weight_added =
          add_layer(cg,
                    make_layer_attrs(projection_weight_attrs),
                    /*inputs=*/{},
                    /*weights=*/{});
      tensor_guid_t t_projection = require_only_key(
          projection_weight_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult bias_weight_added =
          add_layer(cg,
                    make_layer_attrs(bias_weight_attrs),
                    /*inputs=*/{},
                    /*weights=*/{});
      tensor_guid_t t_bias =
          require_only_key(bias_weight_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult linear_added = add_layer(cg,
                                                make_layer_attrs(linear_attrs),
                                                /*inputs=*/
                                                {
                                                    {
                                                        TensorSlotName::INPUT,
                                                        t_input,
                                                    },
                                                },
                                                /*weights=*/
                                                {
                                                    {
                                                        TensorSlotName::WEIGHT,
                                                        t_projection,
                                                    },
                                                    {
                                                        TensorSlotName::BIAS,
                                                        t_bias,
                                                    },
                                                });
      tensor_guid_t t_linear =
          require_only_key(linear_added.outputs, TensorSlotName::OUTPUT);

      add_layer(cg,
                make_layer_attrs(make_relu_attrs()),
                /*inputs=*/
                {
                    {
                        TensorSlotName::INPUT,
                        t_linear,
                    },
                },
                /*weights=*/{});

      return cg;
    }();

    ParallelComputationGraph correct = [&] {
      auto make_layer_attrs = [](auto const &op_attrs) {
        return ParallelLayerAttrs{
            /*op_attrs=*/PCGOperatorAttrs{op_attrs},
            /*name=*/std::nullopt,
        };
      };

      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input =
          require_only_key(input_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult projection_weight_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(projection_weight_attrs),
                             /*inputs=*/{},
                             /*weights=*/{});
      parallel_tensor_guid_t t_projection = require_only_key(
          projection_weight_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult bias_weight_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(bias_weight_attrs),
                             /*inputs=*/{},
                             /*weights=*/{});
      parallel_tensor_guid_t t_bias =
          require_only_key(bias_weight_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult linear_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(linear_attrs),
                             /*inputs=*/
                             {
                                 {
                                     TensorSlotName::INPUT,
                                     t_input,
                                 },
                             },
                             /*weights=*/
                             {
                                 {
                                     TensorSlotName::WEIGHT,
                                     t_projection,
                                 },
                                 {
                                     TensorSlotName::BIAS,
                                     t_bias,
                                 },
                             });
      parallel_tensor_guid_t t_linear =
          require_only_key(linear_added.outputs, TensorSlotName::OUTPUT);

      add_parallel_layer(pcg,
                         make_layer_attrs(make_relu_attrs()),
                         /*inputs=*/
                         {
                             {
                                 TensorSlotName::INPUT,
                                 t_linear,
                             },
                         },
                         /*weights=*/{});
      return pcg;
    }();

    ParallelComputationGraph result = pcg_from_computation_graph(cg);

    CHECK(pcgs_are_isomorphic(result, correct));
  }
}
