#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "test/utils/rapidcheck.h"
#include "utils/containers/get_only.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("topological_ordering") {
    // TODO(@lockshaw) should probably be replaced with a rapidcheck test that
    // compares ParallelComputationGraph to DataflowGraph, but since we
    // currently don't have rapidcheck generation for DataflowGraph this will
    // have to do for now

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    ParallelLayerAttrs layer_label = some<ParallelLayerAttrs>();
    ParallelTensorAttrs tensor_label = some<ParallelTensorAttrs>();

    ParallelLayerAddedResult layer1_added =
        add_parallel_layer(pcg, layer_label, {}, {tensor_label});
    parallel_layer_guid_t layer1 = layer1_added.parallel_layer;
    parallel_tensor_guid_t tensor1 = get_only(layer1_added.outputs);

    ParallelLayerAddedResult layer2_added =
        add_parallel_layer(pcg, layer_label, {tensor1}, {tensor_label});
    parallel_layer_guid_t layer2 = layer2_added.parallel_layer;
    parallel_tensor_guid_t tensor2 = get_only(layer2_added.outputs);

    ParallelLayerAddedResult layer3_added =
        add_parallel_layer(pcg, layer_label, {tensor2}, {tensor_label});
    parallel_layer_guid_t layer3 = layer3_added.parallel_layer;
    parallel_tensor_guid_t tensor3 = get_only(layer3_added.outputs);

    std::vector<parallel_layer_guid_t> result = topological_ordering(pcg);
    // std::vector<parallel_layer_guid_t> correct = {layer1, layer2, layer3};
    // CHECK(result == correct);
  }

  TEST_CASE("get_incoming_inputs(ParallelComputationGraph, parallel_layer_guid_t)") {
    ParallelTensorShape input_shape = ParallelTensorShape{
      ParallelTensorDims{
        FFOrdered<ShardParallelDim>{
          ShardParallelDim{10, 2},
          ShardParallelDim{12, 1},
        },
        ReplicaParallelDimSet{
          SumDegree{1},
          DiscardCopyDegree{1},
        },
      },
      DataType::FLOAT,
    };

    SUBCASE("layer has no inputs") {
      std::string input_name = "my input";
      ParallelComputationGraph pcg = [&] {
        ParallelComputationGraphBuilder b;

        b.create_input_tensor(input_shape, CreateGrad::YES, input_name);

        return b.pcg;
      }();

      parallel_layer_guid_t input_layer = get_parallel_layer_by_name(pcg, input_name);

      std::vector<parallel_tensor_guid_t> result = get_incoming_inputs(pcg, input_layer);
      std::vector<parallel_tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights") {
      std::string my_op_name = "my op";

      ParallelComputationGraphBuilder b;

      parallel_tensor_guid_t input = b.create_input_tensor(input_shape, CreateGrad::YES);
      b.dense(input, 
              /*outDim=*/14, 
              /*activation=*/Activation::RELU,
              /*use_bias=*/true,
              /*data_type=*/DataType::FLOAT,
              /*projection_initializer=*/std::nullopt,
              /*bias_initializer=*/std::nullopt,
              /*name=*/my_op_name);

      ParallelComputationGraph pcg = b.pcg;

      parallel_layer_guid_t my_op_layer = get_parallel_layer_by_name(pcg, my_op_name);

      std::vector<parallel_tensor_guid_t> result = get_incoming_inputs(pcg, my_op_layer);
      std::vector<parallel_tensor_guid_t> correct = {input};

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_incoming_weights(ParallelComputationGraph, parallel_layer_guid_t)") {
    ParallelTensorShape input_shape = ParallelTensorShape{
      ParallelTensorDims{
        FFOrdered<ShardParallelDim>{
          ShardParallelDim{10, 2},
          ShardParallelDim{12, 1},
        },
        ReplicaParallelDimSet{
          SumDegree{1},
          DiscardCopyDegree{1},
        },
      },
      DataType::FLOAT,
    };

    SUBCASE("layer has no inputs or weights") {
      std::string input_name = "my input";
      ParallelComputationGraph pcg = [&] {
        ParallelComputationGraphBuilder b;

        b.create_input_tensor(input_shape, CreateGrad::YES, input_name);

        return b.pcg;
      }();

      parallel_layer_guid_t input_layer = get_parallel_layer_by_name(pcg, input_name);

      std::vector<parallel_tensor_guid_t> result = get_incoming_weights(pcg, input_layer);
      std::vector<parallel_tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs but no weights") {
      std::string my_op_name = "my op";
      ParallelComputationGraph pcg = [&] {
        ParallelComputationGraphBuilder b;

        parallel_tensor_guid_t input = b.create_input_tensor(input_shape, CreateGrad::YES);
        b.relu(input, my_op_name);

        return b.pcg;
      }();

      parallel_layer_guid_t my_op_layer = get_parallel_layer_by_name(pcg, my_op_name);

      std::vector<parallel_tensor_guid_t> result = get_incoming_weights(pcg, my_op_layer);
      std::vector<parallel_tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights, and weights are separate by parallel ops") {
      std::string my_op_name = "my op";

      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      LinearAttrs op_attrs = LinearAttrs{
        /*out_channels=*/14,
        /*use_bias=*/false,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/Activation::RELU,
        /*regularizer=*/std::nullopt,
      };

      ParallelLayerAddedResult input_added = [&] {
        ParallelLayerAttrs input_attrs = ParallelLayerAttrs{
          PCGOperatorAttrs{InputAttrs{}},
          std::nullopt,
        };
        ParallelTensorAttrs input_tensor_attrs = ParallelTensorAttrs{
          input_shape,
          /*sync_type=*/std::nullopt,
          /*initializer=*/std::nullopt,
          CreateGrad::YES
        };

        return add_parallel_layer(pcg, input_attrs, {}, {input_tensor_attrs});
      }();
      parallel_tensor_guid_t input = get_only(input_added.outputs);

      ParallelLayerAddedResult projection_weight_added = [&] {
        ParallelTensorShape projection_weight_shape = throw_if_unexpected(get_projection_shape(op_attrs, input_shape));

        TensorShape unpar_projection_shape = get_reduced_shape(projection_weight_shape);
        ParallelTensorShape raw_projection_weight_shape = lift_to_parallel(unpar_projection_shape);

        ParallelLayerAttrs raw_projection_weight_attrs = ParallelLayerAttrs{
          PCGOperatorAttrs{WeightAttrs{unpar_projection_shape}},
          std::nullopt,
        };
        ParallelTensorAttrs raw_projection_tensor_attrs = ParallelTensorAttrs{
          raw_projection_weight_shape,
          /*sync_type=*/std::nullopt,
          /*initializer=*/std::nullopt,
          CreateGrad::YES
        };

        ParallelLayerAddedResult raw_weight_added = add_parallel_layer(pcg, raw_projection_weight_attrs, {}, {raw_projection_tensor_attrs});

        ReplicateAttrs replicate_attrs = ReplicateAttrs{/*degree=*/2};
        ParallelLayerAttrs replicate_layer_attrs = ParallelLayerAttrs{
          PCGOperatorAttrs{replicate_attrs},
          std::nullopt,
        };
        ParallelTensorAttrs replicated_projection_tensor_attrs = ParallelTensorAttrs{
          get_output_shape(replicate_attrs, raw_projection_weight_shape),
          /*sync_type=*/std::nullopt,
          /*initializer=*/std::nullopt,
          CreateGrad::YES
        };
        return add_parallel_layer(pcg, replicate_layer_attrs, {}, {replicated_projection_tensor_attrs});
      }();
      parallel_tensor_guid_t projection_weight = get_only(projection_weight_added.outputs);

      ParallelLayerAddedResult my_op_added = [&] {
        ParallelTensorShape output_shape = throw_if_unexpected(get_output_shape(op_attrs, input_shape));

        ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
          PCGOperatorAttrs{op_attrs},
          std::nullopt,
        };
        ParallelTensorAttrs output_tensor_attrs = ParallelTensorAttrs{
          output_shape,
          /*sync_type=*/std::nullopt,
          /*initializer=*/std::nullopt,
          CreateGrad::YES
        };

        return add_parallel_layer(pcg, layer_attrs, {input, projection_weight}, {output_tensor_attrs});
      }();

      parallel_layer_guid_t my_op_layer = my_op_added.parallel_layer;

      std::vector<parallel_tensor_guid_t> result = get_incoming_weights(pcg, my_op_layer);
      std::vector<parallel_tensor_guid_t> correct = {projection_weight};

      CHECK(result == correct);
    }
  }
}
