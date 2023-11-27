#include "doctest.h"
#include "pcg/file_format/v1/operator_attrs.h"
#include "utils.h"
#include "utils/containers.h"
#include "utils/json.h"
#include "utils/required.h"

using namespace FlexFlow;

// FIXME: Check deserialization as well. This is currently not implemented
// because of a bug that prevents req from being properly deserialized.
//
// The comments below may apply to multiple test cases.
//
// The checks for the attributes compare the number of fields to ensure that if
// a field is added/removed, the check fails since it may be necessary to update
// the test in that case.
//
// Floating point numbers may be serialized with additional digits. Just check
// that the digits that were provided in the initialization are present. Is it
// even guaranteed that those digits will appear in the serialized result?

TEST_CASE("AggregateAttrs") {
  AggregateAttrs a = {42, 3.14};
  V1AggregateAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"lambda_bal", "3.14"}, {"n", "42"}});
  // TODO: Check deserialization.
}

TEST_CASE("AggregateSpec") {
  AggregateSpecAttrs a = {42, 3.14};
  V1AggregateSpecAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"lambda_bal", "3.14"}, {"n", "42"}});
  // TODO: Check deserialization.
}

TEST_CASE("MultiHeadAttentionAttrs") {
  MultiHeadAttentionAttrs a = {1, 2, 3, 4, 5.67, false, true, false};
  V1MultiHeadAttentionAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j,
               {{"embed_dim", "1"},
                {"num_heads", "2"},
                {"kdim", "3"},
                {"vdim", "4"},
                {"dropout", "5.67"},
                {"bias", "false"},
                {"add_bias_kv", "true"},
                {"add_zero_attn", "false"}});
  // TODO: Check deserialization.
}

TEST_CASE("BatchMatmulAttrs") {
  BatchMatmulAttrs a = {12, 34};
  V1BatchMatmulAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"a_seq_length_dim", "12"}, {"b_seq_length_dim", "34"}});
  // TODO: Check deserialization.
}

TEST_CASE("BatchNormAttrs") {
  BatchNormAttrs a = {true};
  V1BatchNormAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"relu", "true"}});
  // TODO: Check deserialization.
}

TEST_CASE("BroadcastAttrs") {
  BroadcastAttrs a = {stack_vector<int, MAX_TENSOR_DIM>()};
  a.target_dims.push_back(1);
  a.target_dims.push_back(2);
  a.target_dims.push_back(3);
  a.target_dims.push_back(4);
  V1BroadcastAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"target_dims", "[1,2,3,4]"}});
  // TODO: Check deserialization.
}

TEST_CASE("CastAttrs") {
  CastAttrs a = {DataType::HALF};
  V1CastAttrs v1 = to_v1(a);

 CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"dtype", "\"HALF\""}});
  // TODO: Check deserialization.
}

TEST_CASE("CombineAttrs") {
  CombineAttrs a = {ff_dim_t(1), 2};
  V1CombineAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"combine_dim", "1"}, {"combine_degree", "2"}});
  // TODO: Check deserialization.
}

TEST_CASE("ConcatAttrs") {
  ConcatAttrs a = {ff_dim_t(43)};
  V1ConcatAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"axis", "43"}});
  // TODO: Check deserialization.
}

TEST_CASE("Conv2DAttrs") {
  Conv2DAttrs a = {1, 2, 3, 4, 5, 6, 7, 8, Activation::SIGMOID, false};
  V1Conv2DAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j,
               {{"out_channels", "1"},
                {"kernel_h", "2"},
                {"kernel_w", "3"},
                {"stride_h", "4"},
                {"stride_w", "5"},
                {"padding_h", "6"},
                {"padding_w", "7"},
                {"groups", "8"},
                {"activation", "\"SIGMOID\""},
                {"use_bias", "false"}});
  // TODO: Check deserialization.
}

TEST_CASE("DropoutAttrs") {
  DropoutAttrs a = {3.14, 9823749238472398ULL};
  V1DropoutAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"rate", "3.14"}, {"seed", "9823749238472398"}});
  // TODO: Check deserialization.
}

TEST_CASE("ElementBinaryAttrs") {
  ElementBinaryAttrs a = {Op::SQUEEZE, DataType::FLOAT, false, true};
  V1ElementBinaryAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(
      j, {{"should_broadcast_lhs", "false"}, {"should_broadcast_rhs", "true"}});
  // TODO: Check deserialization.
}

TEST_CASE("ElementUnaryAttrs") {
  ElementUnaryAttrs a = {Op::LOGICAL_NOT};
  V1ElementUnaryAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"op", "\"LOGICAL_NOT\""}});
  // TODO: Check deserialization.
}

TEST_CASE("ElementUnaryScalarAttrs") {
  ElementScalarUnaryAttrs a = {Op::SCALAR_FLOOR_DIV, 2.71828};
  V1ElementScalarUnaryAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"op", "\"SCALAR_FLOOR_DIV\""}, {"scalar", "2.71828"}});
  // TODO: Check deserialization.
}

TEST_CASE("AggregateOp") {
  V1AggregateOp v1Sum = to_v1(AggregateOp::SUM);
  CHECK(from_v1(v1Sum) == AggregateOp::SUM);
  CHECK(str(json(v1Sum)) == "\"SUM\"");

  V1AggregateOp v1Avg = to_v1(AggregateOp::AVG);
  CHECK(from_v1(v1Avg) == AggregateOp::AVG);
  CHECK(str(json(v1Avg)) == "\"AVG\"");
}

TEST_CASE("EmbeddingAttrs") {
  EmbeddingAttrs a = {1, 2, AggregateOp::SUM, DataType::DOUBLE};
  V1EmbeddingAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j,
               {{"num_entries", "1"},
                {"out_channels", "2"},
                {"aggr", "\"SUM\""},
                {"data_type", "\"DOUBLE\""}});
  // TODO: Check deserialization.
}

TEST_CASE("FlatAttrs") {
  FlatAttrs a;
  V1FlatAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {});
  // TODO: Check deserialization.
}

TEST_CASE("GatherAttrs") {
  GatherAttrs a = {ff_dim_t(42)};
  V1GatherAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"dim", "42"}});
  // TODO: Check deserialization.
}

TEST_CASE("Group_byAttrs") {
  Group_byAttrs a = {11, 3.14};
  V1Group_byAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"n", "11"}, {"alpha", "3.14"}});
  // TODO: Check deserialization.
}

TEST_CASE("InputAttrs") {
  InputAttrs a;
  V1InputAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {});
  // TODO: Check deserialization.
}

TEST_CASE("LayerNormAttrs") {
  LayerNormAttrs a = {stack_vector<ff_dim_t, MAX_TENSOR_DIM>(), false, 2.71828};
  a.axes.push_back(ff_dim_t(19));
  a.axes.push_back(ff_dim_t(29));
  a.axes.push_back(ff_dim_t(39));
  V1LayerNormAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j,
               {{"axes", "[19,29,39]"},
                {"elementwise_affine", "false"},
                {"eps", "2.71828"}});
  // TODO: Check deserialization.
}

TEST_CASE("L1RegularizerAttrs") {
  L1RegularizerAttrs a = {3.14159};
  V1L1RegularizerAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"lambda", "3.14159"}});
  // TODO: Check deserialization.
}

TEST_CASE("L2RegularizerAttrs") {
  L2RegularizerAttrs a = {3.14159};
  V1L2RegularizerAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"lambda", "3.14159"}});
  // TODO: Check deserialization.
}

TEST_CASE("LinearAttrs") {
  L1RegularizerAttrs r = {1234.567};
  LinearAttrs a = {11, false, DataType::HALF, Activation::TANH, r};
  V1LinearAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j,
               {{"out_channels", "11"},
                {"use_bias", "false"},
                {"data_type", "\"HALF\""},
                {"activation", "\"TANH\""},
                {"regularizer",
                 "{\"index\":0,\"type\":\"::FlexFlow::V1L1RegularizerAttrs\","
                 "\"value\":{\"lambda\":1234.567"}});
  // TODO: Check deserialization.
}

TEST_CASE("NoopAttrs") {
  NoopAttrs a;
  V1NoopAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {});
  // TODO: Check deserialization.
}

TEST_CASE("PoolOp") {
  V1PoolOp v1Max = to_v1(PoolOp::MAX);
  CHECK(from_v1(v1Max) == PoolOp::MAX);
  CHECK(str(json(v1Max)) == "\"MAX\"");

  V1PoolOp v1Avg = to_v1(PoolOp::AVG);
  CHECK(from_v1(v1Avg) == PoolOp::AVG);
  CHECK(str(json(v1Avg)) == "\"AVG\"");
}

TEST_CASE("Pool2DAttrs") {
  Pool2DAttrs a = {1, 2, 3, 4, 5, 6, PoolOp::MAX, Activation::RELU};
  V1Pool2DAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j,
               {{"kernel_h", "1"},
                {"kernel_w", "2"},
                {"stride_h", "3"},
                {"stride_w", "4"},
                {"padding_h", "5"},
                {"padding_w", "6"},
                {"pool_type", "\"MAX\""},
                {"activation", "\"RELU\""}});
  // TODO: Check deserialization.
}

TEST_CASE("ReduceAttrs") {
  ReduceAttrs a = {
      stack_vector<ff_dim_t, MAX_TENSOR_DIM>(), Op::LEAKYRELU, true};
  a.axes.push_back(ff_dim_t(19));
  a.axes.push_back(ff_dim_t(29));
  a.axes.push_back(ff_dim_t(39));
  V1ReduceAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j,
               {{"axes", "[19,29,39]"},
                {"op_type", "\"LEAKYRELU\""},
                {"keepdims", "true"}});
  // TODO: Check deserialization.
}

TEST_CASE("ReductionAttrs") {
  ReductionAttrs a = {ff_dim_t(66), ff_dim_t(77)};
  V1ReductionAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"reduction_dim", "66"}, {"reduction_degree", "77"}});
  // TODO: Check deserialization.
}

TEST_CASE("RepartitionAttrs") {
  RepartitionAttrs a = {ff_dim_t(66), ff_dim_t(77)};
  V1RepartitionAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"repartition_dim", "66"}, {"repartition_degree", "77"}});
  // TODO: Check deserialization.
}

TEST_CASE("ReplicateAttrs") {
  ReplicateAttrs a = {ff_dim_t(66), ff_dim_t(77)};
  V1ReplicateAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"replicate_dim", "66"}, {"replicate_degree", "77"}});
  // TODO: Check deserialization.
}

TEST_CASE("ReshapeAttrs") {
  // TODO: IMPLEMENT THIS.
}

TEST_CASE("ReverseAttrs") {
  ReverseAttrs a = {ff_dim_t(11)};
  V1ReverseAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"axis", "11"}});
  // TODO: Check deserialization.
}

TEST_CASE("SoftmaxAttrs") {
  SoftmaxAttrs a = {ff_dim_t(37)};
  V1SoftmaxAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"dim", "37"}});
  // TODO: Check deserialization.
}

TEST_CASE("SplitAttrs") {
  SplitAttrs a = {stack_vector<int, MAX_NUM_OUTPUTS>(), ff_dim_t(97)};
  a.splits.push_back(53);
  a.splits.push_back(67);
  V1SplitAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"splits", "[53,67]"}, {"axis", "97"}});
  // TODO: Check deserialization.
}

TEST_CASE("TopKAttrs") {
  TopKAttrs a = {17, true};
  V1TopKAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"k", "17"}, {"sorted", "true"}});
  // TODO: Check deserialization.
}

TEST_CASE("TransposeAttrs") {
  TransposeAttrs a = {stack_vector<ff_dim_t, MAX_TENSOR_DIM>()};
  a.perm.push_back(ff_dim_t(3));
  a.perm.push_back(ff_dim_t(43));
  V1TransposeAttrs v1 = to_v1(a);

  CHECK(from_v1(v1) == a);

  json j = v1;
  check_fields(j, {{"perm", "[3,43]"}});
  // TODO: Check deserialization.
}
