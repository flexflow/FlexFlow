#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"
#include "utils/containers/count.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/items.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/containers/without_nullopts.h"
#include "utils/hash/pair.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

/**
 * Stylistically these tests are not great (they're rather complicated
 * and hard to read) and should not be used as a model for other FlexFlow
 * tests.
 *
 * Improving them is being tracked in
 * https://github.com/flexflow/FlexFlow/issues/1474
 */
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ParallelComputationGraphBuilder::add") {
    ParallelComputationGraphBuilder b;

    ShardParallelDim d1 = ShardParallelDim{10_p, 2_p};
    ShardParallelDim d2 = ShardParallelDim{15_p, 3_p};

    TensorShape lhs_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                15_p,
            },
        },
        DataType::FLOAT,
    };

    TensorShape rhs_shape = lhs_shape;

    parallel_tensor_guid_t lhs = b.create_input_tensor(lhs_shape);
    parallel_tensor_guid_t rhs = b.create_input_tensor(rhs_shape);

    parallel_tensor_guid_t out = b.add(lhs, rhs);
    parallel_layer_guid_t layer = get_source_layer(out);

    SUBCASE("incoming") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::LHS_INPUT,
              lhs,
          },
          {
              TensorSlotName::RHS_INPUT,
              rhs,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_outgoing_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::OUTPUT,
              out,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("op attrs") {
      PCGOperatorAttrs result = get_parallel_layer_attrs(b.pcg, layer).op_attrs;
      PCGOperatorAttrs correct = PCGOperatorAttrs{
          ElementBinaryAttrs{
              /*op=*/ElementBinaryOp::ADD,
              /*compute_type=*/DataType::FLOAT,
              /*should_broadcast_lhs=*/false,
              /*should_broadcast_rhs=*/false,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::cast") {
    ParallelComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                12_p,
            },
        },
        DataType::FLOAT,
    };

    DataType output_datatype = DataType::DOUBLE;
    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output = b.cast(input, output_datatype);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::INPUT,
              input,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_outgoing_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::OUTPUT,
              output,
          },
      };
      CHECK(result == correct);

      ParallelTensorShape output_shape =
          get_parallel_tensor_attrs(b.pcg, output).shape;
      CHECK(output_shape.data_type == output_datatype);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::conv2d") {
    ParallelComputationGraphBuilder b;

    positive_int batch_size = 2_p;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, 3_p, 10_p, 10_p}},
        DataType::FLOAT,
    };

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t par_input =
        b.parallel_partition(input, ff_dim_t{0_n}, 2_p);

    ParallelTensorShape par_input_shape = b.get_shape(par_input);

    positive_int outChannels = 6_p;
    positive_int kernelH = 5_p;
    positive_int kernelW = 4_p;
    positive_int strideH = 3_p;
    positive_int strideW = 2_p;
    nonnegative_int paddingH = 1_n;
    nonnegative_int paddingW = 0_n;
    parallel_tensor_guid_t output = b.conv2d(par_input,
                                             /*outChannels=*/outChannels,
                                             /*kernelH=*/kernelH,
                                             /*kernelW=*/kernelW,
                                             /*strideH=*/strideH,
                                             /*strideW=*/strideW,
                                             /*paddingH=*/paddingH,
                                             /*paddingW=*/paddingW);

    std::map<parallel_layer_guid_t, ParallelLayerAttrs> layers = generate_map(
        pcg_get_parallel_layers(b.pcg), [&](parallel_layer_guid_t const &l) {
          return get_parallel_layer_attrs(b.pcg, l);
        });
    CHECK_MESSAGE(layers.size() == 7, "Incorrect layers ", layers);

    auto num_attrs_of_type = [&](OperatorType op_type) -> nonnegative_int {
      return count(values(layers), [&](ParallelLayerAttrs const &l) {
        return get_op_type(l) == op_type;
      });
    };

    nonnegative_int num_weight_attrs = num_attrs_of_type(OperatorType::WEIGHT);
    CHECK(num_weight_attrs == 2);

    nonnegative_int num_input_attrs = num_attrs_of_type(OperatorType::INPUT);
    CHECK(num_input_attrs == 1);

    nonnegative_int num_conv_attrs = num_attrs_of_type(OperatorType::CONV2D);
    CHECK(num_conv_attrs == 1);

    nonnegative_int num_replicate_attrs =
        num_attrs_of_type(OperatorType::REPLICATE);
    CHECK(num_replicate_attrs == 2);

    nonnegative_int num_partition_attrs =
        num_attrs_of_type(OperatorType::REPARTITION);
    CHECK(num_partition_attrs == 1);

    parallel_layer_guid_t conv_guid = get_only(without_nullopts(transform(
        vector_of(items(layers)),
        [](std::pair<parallel_layer_guid_t, ParallelLayerAttrs> const &kv)
            -> std::optional<parallel_layer_guid_t> {
          if (get_op_type(kv.second) == OperatorType::CONV2D) {
            return kv.first;
          } else {
            return std::nullopt;
          }
        })));
    Conv2DAttrs conv_attrs = layers.at(conv_guid).op_attrs.get<Conv2DAttrs>();
    Conv2DAttrs correct_attrs = Conv2DAttrs{
        outChannels,
        kernelH,
        kernelW,
        strideH,
        strideW,
        paddingH,
        paddingW,
        /*groups=*/1_p,
        /*activation=*/std::nullopt,
        /*use_bias=*/true,
    };
    CHECK(conv_attrs == correct_attrs);

    ParallelTensorShape correct_output_shape =
        conv2d_get_output_parallel_shape(correct_attrs, par_input_shape);
    ParallelTensorShape correct_kernel_shape =
        conv2d_get_kernel_parallel_shape(correct_attrs, par_input_shape);
    ParallelTensorShape correct_bias_shape =
        conv2d_get_bias_parallel_shape(correct_attrs, par_input_shape);

    std::map<TensorSlotName, parallel_tensor_guid_t> conv_incoming =
        get_incoming_tensors(b.pcg, conv_guid);

    parallel_tensor_guid_t conv_input = conv_incoming.at(TensorSlotName::INPUT);
    ParallelTensorShape conv_input_shape =
        get_parallel_tensor_attrs(b.pcg, conv_input).shape;
    CHECK(conv_input_shape == par_input_shape);

    parallel_tensor_guid_t conv_kernel =
        conv_incoming.at(TensorSlotName::FILTER);
    ParallelTensorShape conv_kernel_shape =
        get_parallel_tensor_attrs(b.pcg, conv_kernel).shape;
    CHECK(conv_kernel_shape == correct_kernel_shape);

    parallel_tensor_guid_t conv_bias = conv_incoming.at(TensorSlotName::BIAS);
    ParallelTensorShape conv_bias_shape =
        get_parallel_tensor_attrs(b.pcg, conv_bias).shape;
    CHECK(conv_bias_shape == correct_bias_shape);

    std::map<TensorSlotName, parallel_tensor_guid_t> conv_outputs =
        get_outgoing_tensors(b.pcg, conv_guid);
    CHECK(conv_outputs.size() == 1);

    parallel_tensor_guid_t conv_output =
        require_only_key(conv_outputs, TensorSlotName::OUTPUT);
    ParallelTensorShape conv_output_shape =
        get_parallel_tensor_attrs(b.pcg, conv_output).shape;
    CHECK(conv_output_shape == correct_output_shape);
  };

  TEST_CASE("ParallelComputationGraphBuilder::dense") {
    ParallelComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                16_p,
            },
        },
        DataType::FLOAT,
    };
    positive_int outDim = 14_p;

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output = b.dense(input,
                                            outDim,
                                            Activation::RELU,
                                            /*use_bias=*/true,
                                            DataType::FLOAT);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      CHECK(result.at(TensorSlotName::INPUT) == input);

      CHECK(result.size() == 3);
    }

    SUBCASE("outputs") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_outgoing_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::OUTPUT,
              output,
          },
      };
      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::embedding") {
    ParallelComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                12_p,
                10_p,
            },
        },
        DataType::INT32,
    };

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output = b.embedding(input,
                                                /*num_entries=*/32_p,
                                                /*outDim=*/8_p,
                                                AggregateOp::SUM,
                                                DataType::FLOAT);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);

      CHECK(result.size() == 2);
      CHECK(result.at(TensorSlotName::INPUT) == input);
    }

    SUBCASE("outputs") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_outgoing_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::OUTPUT,
              output,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::multihead_attention") {
    ParallelComputationGraphBuilder b;

    TensorShape query_shape = TensorShape{
        TensorDims{
            FFOrdered{
                12_p,
                16_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    TensorShape key_shape = query_shape;
    TensorShape value_shape = query_shape;

    positive_int embed_dim = 8_p;
    positive_int num_heads = 6_p;

    parallel_tensor_guid_t query = b.create_input_tensor(query_shape);
    parallel_tensor_guid_t key = b.create_input_tensor(key_shape);
    parallel_tensor_guid_t value = b.create_input_tensor(value_shape);
    parallel_tensor_guid_t output =
        b.multihead_attention(query, key, value, embed_dim, num_heads);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);

      CHECK(result.size() == 6);
      CHECK(result.at(TensorSlotName::QUERY) == query);
      CHECK(result.at(TensorSlotName::KEY) == key);
      CHECK(result.at(TensorSlotName::VALUE) == value);
    }

    SUBCASE("outputs") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_outgoing_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::OUTPUT,
              output,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::relu") {
    ParallelComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                18_p,
                32_p,
            },
        },
        DataType::FLOAT,
    };

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output = b.relu(input);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::INPUT,
              input,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_outgoing_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::OUTPUT,
              output,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::parallel_partition") {
    ParallelComputationGraphBuilder b;

    ShardParallelDim batch_dim = ShardParallelDim{18_p, 2_p};
    ShardParallelDim feature_dim = ShardParallelDim{10_p, 1_p};

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                18_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output =
        b.parallel_partition(input, ff_dim_t{0_n}, 2_p);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::INPUT,
              input,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_outgoing_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::OUTPUT,
              output,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::parallel_combine") {
    ParallelComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                18_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    input = b.parallel_partition(input, ff_dim_t{0_n}, 2_p);
    parallel_tensor_guid_t output =
        b.parallel_combine(input, ff_dim_t{0_n}, 2_p);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::INPUT,
              input,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_outgoing_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::OUTPUT,
              output,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::parallel_replicate") {
    ParallelComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                18_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output = b.parallel_replicate(input, 2_p);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::INPUT,
              input,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_outgoing_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::OUTPUT,
              output,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::parallel_reduce") {
    ParallelComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                18_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    input = b.parallel_partition(input, ff_dim_t{1_n}, 2_p);
    input = b.dense(input,
                    /*out_dim=*/12_p,
                    /*activation=*/std::nullopt,
                    /*use_bias=*/false);
    parallel_tensor_guid_t output = b.parallel_reduce(input, 2_p);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::INPUT,
              input,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::map<TensorSlotName, parallel_tensor_guid_t> result =
          get_outgoing_tensors(b.pcg, layer);
      std::map<TensorSlotName, parallel_tensor_guid_t> correct = {
          {
              TensorSlotName::OUTPUT,
              output,
          },
      };

      CHECK(result == correct);
    }
  }
}
