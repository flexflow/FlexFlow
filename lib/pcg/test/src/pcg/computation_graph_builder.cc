#include "pcg/computation_graph_builder.h"
#include "op-attrs/ops/split.h"
#include "pcg/computation_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ComputationGraphBuilder::batch_norm") {
    ComputationGraphBuilder b;

    positive_int batch_size = 4_p;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, 3_p, 10_p, 12_p}},
        DataType::FLOAT,
    };

    tensor_guid_t input = b.create_input(input_shape, CreateGrad::NO);
    tensor_guid_t result = b.batch_norm(input);

    SUBCASE("output tensor shape is correct") {
      TensorShape result_output_shape = b.get_shape(result);

      TensorShape correct_output_shape = input_shape;

      CHECK(result_output_shape == correct_output_shape);
    }

    SUBCASE("computation graph contains correct operator counts") {
      std::map<OperatorType, positive_int> result_op_type_counts =
          operator_type_counts_in_computation_graph(b.computation_graph);

      std::map<OperatorType, positive_int> correct_op_type_counts = {
          {OperatorType::INPUT, 1_p},
          {OperatorType::WEIGHT, 2_p},
          {OperatorType::BATCHNORM, 1_p},
      };

      CHECK(result_op_type_counts == correct_op_type_counts);
    }
  }

  TEST_CASE("ComputationGraphBuilder::conv2d") {
    ComputationGraphBuilder b;

    positive_int batch_size = 2_p;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, 3_p, 10_p, 10_p}},
        DataType::FLOAT,
    };

    tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);

    SUBCASE("without bias") {
      tensor_guid_t result = b.conv2d(input,
                                      /*outChannels=*/5_p,
                                      /*kernelH=*/3_p,
                                      /*kernelW=*/3_p,
                                      /*strideH=*/1_p,
                                      /*strideW=*/1_p,
                                      /*paddingH=*/0_n,
                                      /*paddingW=*/0_n,
                                      /*activation=*/std::nullopt,
                                      /*groups=*/1_p,
                                      /*use_bias=*/false);

      SUBCASE("output tensor shape is correct") {
        TensorShape result_output_shape = b.get_shape(result);

        TensorShape correct_output_shape = TensorShape{
            TensorDims{FFOrdered{batch_size, 5_p, 8_p, 8_p}},
            DataType::FLOAT,
        };

        CHECK(result_output_shape == correct_output_shape);
      }

      SUBCASE("computation graph contains correct operator counts") {
        std::map<OperatorType, positive_int> result_op_type_counts =
            operator_type_counts_in_computation_graph(b.computation_graph);

        std::map<OperatorType, positive_int> correct_op_type_counts = {
            {OperatorType::INPUT, 1_p},
            {OperatorType::CONV2D, 1_p},
            {OperatorType::WEIGHT, 1_p},
        };

        CHECK(result_op_type_counts == correct_op_type_counts);
      }
    }

    SUBCASE("with bias") {
      tensor_guid_t result = b.conv2d(input,
                                      /*outChannels=*/5_p,
                                      /*kernelH=*/3_p,
                                      /*kernelW=*/3_p,
                                      /*strideH=*/1_p,
                                      /*strideW=*/1_p,
                                      /*paddingH=*/0_n,
                                      /*paddingW=*/0_n,
                                      /*activation=*/std::nullopt,
                                      /*groups=*/1_p,
                                      /*use_bias=*/true);

      SUBCASE("output tensor shape is correct") {
        TensorShape result_output_shape = b.get_shape(result);

        TensorShape correct_output_shape = TensorShape{
            TensorDims{FFOrdered{batch_size, 5_p, 8_p, 8_p}},
            DataType::FLOAT,
        };

        CHECK(result_output_shape == correct_output_shape);
      }

      SUBCASE("computation graph contains correct operator counts") {
        std::map<OperatorType, positive_int> result_op_type_counts =
            operator_type_counts_in_computation_graph(b.computation_graph);

        std::map<OperatorType, positive_int> correct_op_type_counts = {
            {OperatorType::INPUT, 1_p},
            {OperatorType::CONV2D, 1_p},
            {OperatorType::WEIGHT, 2_p},
        };

        CHECK(result_op_type_counts == correct_op_type_counts);
      }
    }
  }

  TEST_CASE("ComputationGraphBuilder::split") {
    ComputationGraphBuilder b;

    positive_int batch_size = 2_p;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, 3_p, 10_p, 10_p}},
        DataType::FLOAT,
    };

    tensor_guid_t input = b.create_input(input_shape, CreateGrad::NO);
    std::vector<tensor_guid_t> result = b.split(
        input, std::vector<positive_int>{2_p, 5_p, 3_p}, relative_ff_dim_t{2});

    SUBCASE("output tensor shapes are correct") {
      std::vector<TensorShape> result_shapes =
          transform(result, [&](tensor_guid_t t) -> TensorShape {
            return b.get_shape(t);
          });

      auto mk_correct_shape = [&](positive_int x) -> TensorShape {
        return TensorShape{
            TensorDims{FFOrdered{batch_size, 3_p, x, 10_p}},
            DataType::FLOAT,
        };
      };

      std::vector<TensorShape> correct_shapes = {
          mk_correct_shape(2_p),
          mk_correct_shape(5_p),
          mk_correct_shape(3_p),
      };

      CHECK(result_shapes == correct_shapes);
    }

    SUBCASE("computation graph contains correct operator counts") {
      std::map<OperatorType, positive_int> result_op_type_counts =
          operator_type_counts_in_computation_graph(b.computation_graph);

      std::map<OperatorType, positive_int> correct_op_type_counts = {
          {OperatorType::INPUT, 1_p},
          {OperatorType::SPLIT, 1_p},
      };

      CHECK(result_op_type_counts == correct_op_type_counts);
    }
  }

  TEST_CASE("ComputationGraphBuilder::split rejects too many outputs") {
    ComputationGraphBuilder b;
    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{2_p, 3_p, 10_p, 10_p}},
        DataType::FLOAT,
    };
    tensor_guid_t input = b.create_input(input_shape, CreateGrad::NO);

    CHECK_THROWS(b.split(
        input, {1_p, 1_p, 1_p, 1_p, 1_p, 1_p, 4_p}, relative_ff_dim_t{2}));
  }

  TEST_CASE("ComputationGraphBuilder transpose") {
    ComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{4_p, 3_p, 10_p, 12_p}},
        DataType::FLOAT,
    };

    tensor_guid_t input = b.create_input(input_shape, CreateGrad::NO);
    tensor_guid_t result = b.transpose(input,
                                       /*perm=*/std::vector<nonnegative_int>{
                                           3_n,
                                           1_n,
                                           0_n,
                                           2_n,
                                       });

    SUBCASE("output tensor shape is correct") {
      TensorShape result_output_shape = b.get_shape(result);

      TensorShape correct_output_shape = TensorShape{
          TensorDims{FFOrdered{12_p, 3_p, 4_p, 10_p}},
          DataType::FLOAT,
      };

      CHECK(result_output_shape == correct_output_shape);
    }

    SUBCASE("computation graph contains correct operator counts") {
      std::map<OperatorType, positive_int> result_op_type_counts =
          operator_type_counts_in_computation_graph(b.computation_graph);

      std::map<OperatorType, positive_int> correct_op_type_counts = {
          {OperatorType::INPUT, 1_p},
          {OperatorType::TRANSPOSE, 1_p},
      };

      CHECK(result_op_type_counts == correct_op_type_counts);
    }
  }

  TEST_CASE("ComputationGraphBuilder::upsample") {
    ComputationGraphBuilder b;

    positive_int batch_size = 4_p;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, 3_p, 10_p, 12_p}},
        DataType::FLOAT,
    };

    tensor_guid_t input = b.create_input(input_shape, CreateGrad::NO);
    tensor_guid_t result = b.upsample(input,
                                      /*scale_factor=*/3_ge2,
                                      /*mode=*/UpsampleMode::NEAREST);

    SUBCASE("output tensor shape is correct") {
      TensorShape result_output_shape = b.get_shape(result);

      TensorShape correct_output_shape = TensorShape{
          TensorDims{FFOrdered{batch_size, 3_p, 30_p, 36_p}},
          DataType::FLOAT,
      };

      CHECK(result_output_shape == correct_output_shape);
    }

    SUBCASE("computation graph contains correct operator counts") {
      std::map<OperatorType, positive_int> result_op_type_counts =
          operator_type_counts_in_computation_graph(b.computation_graph);

      std::map<OperatorType, positive_int> correct_op_type_counts = {
          {OperatorType::INPUT, 1_p},
          {OperatorType::UPSAMPLE, 1_p},
      };

      CHECK(result_op_type_counts == correct_op_type_counts);
    }
  }
}
