#include "doctest.h"
#include "pcg/file_format/v1/operator_attrs.h"
#include "utils/containers.h"
#include "utils/json.h"
#include "utils/required.h"

using namespace FlexFlow;

using Field = std::pair<std::string, std::string>;
static void check_fields(json const &j, std::vector<Field> const &fields) {
  std::stringstream ss;
  ss << j;
  std::string strj = ss.str();

  for (auto const &[key, val] : fields) {
    std::stringstream fs;
    fs << "\"" << key << "\":" << val;
    std::string field = fs.str();

    CHECK(strj.find(field) != std::string::npos);
  }
}

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

  CHECK(visit_struct::field_count<AggregateAttrs>() == 2);
  CHECK(visit_struct::field_count<V1AggregateAttrs>() == 2);

  json j = v1;
  check_fields(j, {{"lambda_bal", "3.14"}, {"n", "42"}});
}

TEST_CASE("AggregateSpec") {
  AggregateSpecAttrs a = {42, 3.14};
  V1AggregateSpecAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<AggregateSpecAttrs>() == 2);
  CHECK(visit_struct::field_count<V1AggregateSpecAttrs>() == 2);

  json j = v1;
  check_fields(j, {{"lambda_bal", "3.14"}, {"n", "42"}});
}

TEST_CASE("MultiHeadAttentionAttrs") {
  MultiHeadAttentionAttrs a = {1, 2, 3, 4, 5.67, false, true, false};
  V1MultiHeadAttentionAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<MultiHeadAttentionAttrs>() == 8);
  CHECK(visit_struct::field_count<V1MultiHeadAttentionAttrs>() == 8);

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
}

TEST_CASE("BatchMatmulAttrs") {
  BatchMatmulAttrs a = {12, 34};
  V1BatchMatmulAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<BatchMatmulAttrs>() == 2);
  CHECK(visit_struct::field_count<V1BatchMatmulAttrs>() == 2);

  json j = v1;
  check_fields(j, {{"a_seq_length_dim", "12"}, {"b_seq_length_dim", "34"}});
}

TEST_CASE("BatchNormAttrs") {
  BatchNormAttrs a = {true};
  V1BatchNormAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<BatchNormAttrs>() == 1);
  CHECK(visit_struct::field_count<V1BatchNormAttrs>() == 1);

  json j = v1;
  check_fields(j, {{"relu", "true"}});
}

TEST_CASE("BroadcastAttrs") {
  BroadcastAttrs a = {stack_vector<int, MAX_TENSOR_DIM>()};
  a.target_dims.push_back(1);
  a.target_dims.push_back(2);
  a.target_dims.push_back(3);
  a.target_dims.push_back(4);
  V1BroadcastAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<BroadcastAttrs>() == 1);
  CHECK(visit_struct::field_count<V1BroadcastAttrs>() == 1);

  json j = v1;
  check_fields(j, {{"target_dims", "[1,2,3,4]"}});
}

TEST_CASE("CastAttrs") {
  CastAttrs a = {DataType::HALF};
  V1CastAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<CastAttrs>() == 1);
  CHECK(visit_struct::field_count<V1CastAttrs>() == 1);

  json j = v1;
  check_fields(j, {{"dtype", "\"HALF\""}});
}

TEST_CASE("CombineAttrs") {
  CombineAttrs a = {ff_dim_t(1), 2};
  V1CombineAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<CombineAttrs>() == 2);
  CHECK(visit_struct::field_count<V1CombineAttrs>() == 2);

  json j = v1;
  check_fields(j, {{"combine_dim", "1"}, {"combine_degree", "2"}});
}

TEST_CASE("ConcatAttrs") {
  ConcatAttrs a = {ff_dim_t(43)};
  V1ConcatAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<ConcatAttrs>() == 1);
  CHECK(visit_struct::field_count<V1ConcatAttrs>() == 1);

  json j = v1;
  check_fields(j, {{"axis", "43"}});
}

TEST_CASE("Conv2DAttrs") {
  Conv2DAttrs a = {1, 2, 3, 4, 5, 6, 7, 8, Activation::SIGMOID, false};
  V1Conv2DAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<Conv2DAttrs>() == 10);
  CHECK(visit_struct::field_count<V1Conv2DAttrs>() == 10);

  json j = v1;
  check_fields(j, {{"out_channels", "1"},
                   {"kernel_h", "2"},
                   {"kernel_w", "3"},
                   {"stride_h", "4"},
                   {"stride_w", "5"},
                   {"padding_h", "6"},
                   {"padding_w", "7"},
                   {"groups", "8"},
                   {"activation", "\"SIGMOID\""},
                   {"use_bias", "false"}});
}

TEST_CASE("DropoutAttrs") {
  DropoutAttrs a = {3.14, 9823749238472398ULL};
  V1DropoutAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<DropoutAttrs>() == 2);
  CHECK(visit_struct::field_count<V1DropoutAttrs>() == 2);

  json j = v1;
  check_fields(j, {{"rate", "3.14"},
                   {"seed", "9823749238472398"}});
}

TEST_CASE("ElementBinaryAttrs") {
  ElementBinaryAttrs a = {Op::SQUEEZE, DataType::FLOAT, false, true};
  V1ElementBinaryAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<ElementBinaryAttrs>() == 4);
  CHECK(visit_struct::field_count<V1ElementBinaryAttrs>() == 4);

  json j = v1;
  check_fields(j, {{"should_broadcast_lhs", "false"},
                   {"should_broadcast_rhs", "true"}});
}

TEST_CASE("ElementUnaryAttrs") {
  ElementUnaryAttrs a = {Op::LOGICAL_NOT};
  V1ElementUnaryAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<ElementUnaryAttrs>() == 1);
  CHECK(visit_struct::field_count<V1ElementUnaryAttrs>() == 1);

  json j = v1;
  check_fields(j, {{"op", "\"LOGICAL_NOT\""}});
}

TEST_CASE("ElementUnaryScalarAttrs") {
  ElementScalarUnaryAttrs a = {Op::SCALAR_FLOOR_DIV, 2.71828};
  V1ElementScalarUnaryAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<ElementScalarUnaryAttrs>() == 2);
  CHECK(visit_struct::field_count<V1ElementScalarUnaryAttrs>() == 2);

  json j = v1;
  check_fields(j, {{"op", "\"SCALAR_FLOOR_DIV\""},
                   {"scalar", "2.71828"}});
}

TEST_CASE("EmbeddingAttrs") {
  EmbeddingAttrs a = {1, 2, AggregateOp::SUM, DataType::DOUBLE};
  V1EmbeddingAttrs v1 = to_v1(a);

  CHECK(visit_struct::field_count<EmbeddingAttrs>() == 4);
  CHECK(visit_struct::field_count<V1EmbeddingAttrs>() == 4);

  json j = v1;
  check_fields(j, {{"num_entries", "1"},
                   {"out_channels", "2"},
                   {"aggr", "\"SUM\""},
                   {"data_type", "\"DOUBLE\""}});
}
