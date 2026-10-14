#include "substitutions/unity_substitution_set.h"
#include "op-attrs/activation.h"
#include "pcg/machine_compute_specification.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/substitution_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/require_only_key.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/positive_int/positive_range.h"
#include "utils/random_utils.h"

namespace FlexFlow {

std::vector<Substitution>
    get_unity_substitution_set(MachineComputeSpecification const &resources) {
  std::vector<Substitution> substitutions;

  positive_int max_tensor_dim = positive_int{MAX_TENSOR_DIM};

  for (positive_int dim : positive_range(1_p, max_tensor_dim + 1_p)) {
    for (positive_int degree = 1_p; degree <= get_num_gpus(resources);
         degree *= 2_p) {
      substitutions.push_back(
          create_replicate_linear_combine(dim, degree, true));
      substitutions.push_back(
          create_replicate_linear_combine(dim, degree, false));
      substitutions.push_back(
          create_partition_linear_combine(dim, degree, true));
      substitutions.push_back(
          create_partition_linear_combine(dim, degree, false));
      substitutions.push_back(create_partition_relu_combine(
          ff_dim_t{dim.nonnegative_int_from_positive_int()}, degree));
      substitutions.push_back(create_partition_add_combine(
          ff_dim_t{dim.nonnegative_int_from_positive_int()}, degree));
      substitutions.push_back(create_partition_attention_combine(dim, degree));
      substitutions.push_back(create_replicate_attention_reduce(dim, degree));
    }
  }

  for (positive_int degree = 1_p; degree <= get_num_gpus(resources);
       degree *= 2_p) {
    substitutions.push_back(create_partition_conv2d_combine(4_p, degree));
  }

  for (positive_int partition_dim : positive_range(1_p, max_tensor_dim + 1_p)) {
    for (positive_int softmax_dim : positive_range(1_p, max_tensor_dim + 1_p)) {
      for (positive_int degree = 1_p; degree <= get_num_gpus(resources);
           degree *= 2_p) {
        if (partition_dim != softmax_dim) {
          substitutions.push_back(create_partition_softmax_combine(
              ff_dim_t{partition_dim.nonnegative_int_from_positive_int()},
              ff_dim_t{softmax_dim.nonnegative_int_from_positive_int()},
              degree));
        }
      }
    }
  }

  substitutions.push_back(create_fuse_linear_activation(Activation::RELU));
  substitutions.push_back(create_fuse_linear_activation(Activation::SIGMOID));
  substitutions.push_back(create_fuse_linear_activation(Activation::TANH));
  substitutions.push_back(create_fuse_linear_activation(Activation::GELU));

  return substitutions;
}

std::vector<Substitution> get_expanded_substitution_set(
    MachineComputeSpecification const &resources) {

  std::vector<Substitution> substitutions =
      get_unity_substitution_set(resources);

  substitutions.push_back(create_fuse_linear_activation(Activation::SILU));

  substitutions.push_back(create_fuse_batch_norm_activation(Activation::RELU));
  substitutions.push_back(
      create_fuse_batch_norm_activation(Activation::SIGMOID));
  substitutions.push_back(create_fuse_batch_norm_activation(Activation::TANH));
  substitutions.push_back(create_fuse_batch_norm_activation(Activation::GELU));
  substitutions.push_back(create_fuse_batch_norm_activation(Activation::SILU));

  return substitutions;
}

static PatternValue insert_single_output_pattern(
    SubstitutionBuilder &b,
    OperatorAttributePattern const &attribute_pattern,
    std::map<TensorSlotName, PatternValue> const &inputs,
    TensorAttributePattern const &output_pattern,
    std::string const &name) {
  return require_only_key(b.add_pattern_node(attribute_pattern,
                                             inputs,
                                             /*output_patterns=*/
                                             {
                                                 {
                                                     TensorSlotName::OUTPUT,
                                                     output_pattern,
                                                 },
                                             },
                                             name),
                          TensorSlotName::OUTPUT);
}

static OutputGraphExprValue insert_single_output_op(
    SubstitutionBuilder &b,
    OutputOperatorAttrsAssignment const &expr,
    std::map<TensorSlotName, OutputGraphExprValue> const &inputs) {
  return require_only_key(
      b.add_output_graph_node(expr, inputs, {TensorSlotName::OUTPUT}),
      TensorSlotName::OUTPUT);
}

static OutputGraphExprValue
    insert_replicate_or_reduce(OperatorType op_type,
                               SubstitutionBuilder &b,
                               positive_int degree,
                               OutputGraphExprValue const &input) {

  ASSERT(op_type == OperatorType::REPLICATE ||
         op_type == OperatorType::REDUCTION);

  OutputOperatorAttrsAssignment replicate_expr = OutputOperatorAttrsAssignment{
      std::nullopt,
      {
          set_op_type_attr(op_type),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                               OperatorAttributeValue{
                                   degree.nonnegative_int_from_positive_int()}),
      }};

  return insert_single_output_op(
      b, replicate_expr, {{TensorSlotName::INPUT, input}});
}

static OutputGraphExprValue
    insert_replicate(SubstitutionBuilder &b,
                     positive_int degree,
                     OutputGraphExprValue const &input) {
  return insert_replicate_or_reduce(OperatorType::REPLICATE, b, degree, input);
}

static OutputGraphExprValue insert_reduce(SubstitutionBuilder &b,
                                          positive_int degree,
                                          OutputGraphExprValue const &input) {
  return insert_replicate_or_reduce(OperatorType::REDUCTION, b, degree, input);
}

static OutputGraphExprValue
    insert_partition_or_combine(OperatorType op_type,
                                SubstitutionBuilder &b,
                                positive_int degree,
                                ff_dim_t dim,
                                OutputGraphExprValue const &input) {

  ASSERT(op_type == OperatorType::REPARTITION ||
         op_type == OperatorType::COMBINE);

  OutputOperatorAttrsAssignment partition_input_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(op_type),
              set_attr_to_constant(
                  OperatorAttributeKey::PARALLEL_DEGREE,
                  OperatorAttributeValue{
                      degree.nonnegative_int_from_positive_int()}),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                   OperatorAttributeValue{dim}),
          }};

  OutputGraphExprValue o_partition_output = insert_single_output_op(
      b, partition_input_expr, {{TensorSlotName::INPUT, input}});

  return o_partition_output;
}

static OutputGraphExprValue
    insert_partition(SubstitutionBuilder &b,
                     positive_int degree,
                     ff_dim_t dim,
                     OutputGraphExprValue const &input) {

  return insert_partition_or_combine(
      OperatorType::REPARTITION, b, degree, dim, input);
}

static OutputGraphExprValue insert_combine(SubstitutionBuilder &b,
                                           positive_int degree,
                                           ff_dim_t dim,
                                           OutputGraphExprValue const &input) {

  return insert_partition_or_combine(
      OperatorType::COMBINE, b, degree, dim, input);
}

Substitution create_replicate_linear_combine(positive_int num_dims,
                                             positive_int degree,
                                             bool use_bias) {
  SubstitutionBuilder b;

  auto [p_input, o_input] = b.add_input(tensor_attribute_pattern_match_all());
  auto [p_weight, o_weight] = b.add_input(tensor_attribute_pattern_match_all());
  std::map<TensorSlotName, PatternValue> p_inputs = {
      {TensorSlotName::INPUT, p_input},
      {TensorSlotName::WEIGHT, p_weight},
  };

  std::optional<OutputGraphExprValue> o_bias = std::nullopt;
  if (use_bias) {
    std::pair<PatternValue, OutputGraphExprValue> bias =
        b.add_input(tensor_attribute_pattern_match_all());
    p_inputs.insert({
        TensorSlotName::BIAS,
        bias.first,
    });
    o_bias = bias.second;
  }

  OperatorAttributePattern linear_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::LINEAR),
      op_attr_key_equals(OperatorAttributeKey::BIAS,
                         OperatorAttributeValue{use_bias}),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
  }};

  std::string linear_name = "linear";
  PatternValue p_linear_output = insert_single_output_pattern(
      b,
      linear_pattern,
      p_inputs,
      /*output_pattern=*/tensor_attr_pattern_require_num_dims(num_dims),
      linear_name);

  OutputGraphExprValue o_replicate_input_output =
      insert_replicate(b, degree, o_input);

  OutputGraphExprValue o_partition_weights_output =
      insert_partition(b, degree, ff_dim_t{0_n}, o_weight);

  std::map<TensorSlotName, OutputGraphExprValue> o_linear_inputs = {
      {
          TensorSlotName::INPUT,
          o_replicate_input_output,
      },
      {
          TensorSlotName::WEIGHT,
          o_partition_weights_output,
      },
  };

  if (use_bias) {
    OutputGraphExprValue o_partition_bias_output =
        insert_partition(b, degree, ff_dim_t{0_n}, o_bias.value());

    o_linear_inputs.insert({
        TensorSlotName::BIAS,
        o_partition_bias_output,
    });
  }

  OutputOperatorAttrsAssignment linear_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named(linear_name),
      {},
  };
  OutputGraphExprValue o_linear_output =
      insert_single_output_op(b, linear_expr, o_linear_inputs);

  ff_dim_t combine_output_dim = ff_dim_t{
      nonnegative_int{num_dims.int_from_positive_int() - 1},
  };
  OutputGraphExprValue o_combine_output =
      insert_combine(b, degree, combine_output_dim, o_linear_output);

  b.equate_outputs(p_linear_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_partition_linear_combine(positive_int num_dims,
                                             positive_int degree,
                                             bool use_bias) {
  SubstitutionBuilder b;

  auto [p_input, o_input] = b.add_input(tensor_attribute_pattern_match_all());
  auto [p_weight, o_weight] = b.add_input(tensor_attribute_pattern_match_all());
  std::map<TensorSlotName, PatternValue> p_inputs = {
      {
          TensorSlotName::INPUT,
          p_input,
      },
      {
          TensorSlotName::WEIGHT,
          p_weight,
      },
  };

  std::optional<OutputGraphExprValue> o_bias = std::nullopt;
  if (use_bias) {
    std::pair<PatternValue, OutputGraphExprValue> bias =
        b.add_input(tensor_attribute_pattern_match_all());
    p_inputs.insert({
        TensorSlotName::BIAS,
        bias.first,
    });
    o_bias = bias.second;
  }

  OperatorAttributePattern linear_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::LINEAR),
      op_attr_key_equals(OperatorAttributeKey::BIAS,
                         OperatorAttributeValue{use_bias}),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
  }};

  std::string linear_name = "linear";
  PatternValue p_linear_output = insert_single_output_pattern(
      b,
      linear_pattern,
      p_inputs,
      /*output_pattern=*/tensor_attr_pattern_require_num_dims(num_dims),
      linear_name);

  OutputGraphExprValue o_partition_input_output =
      insert_partition(b, degree, ff_dim_t{0_n}, o_input);

  OutputGraphExprValue o_replicate_weights_output =
      insert_replicate(b, degree, o_weight);

  std::map<TensorSlotName, OutputGraphExprValue> o_linear_inputs = {
      {
          TensorSlotName::INPUT,
          o_partition_input_output,
      },
      {
          TensorSlotName::WEIGHT,
          o_replicate_weights_output,
      },
  };

  if (use_bias) {
    OutputGraphExprValue o_replicate_bias_output =
        insert_replicate(b, degree, o_bias.value());

    o_linear_inputs.insert({
        TensorSlotName::BIAS,
        o_replicate_bias_output,
    });
  }

  OutputOperatorAttrsAssignment linear_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named(linear_name),
      {},
  };
  OutputGraphExprValue o_linear_output =
      insert_single_output_op(b, linear_expr, o_linear_inputs);

  ff_dim_t combine_output_dim = ff_dim_t{0_n};

  OutputGraphExprValue o_combine_output =
      insert_combine(b, degree, combine_output_dim, o_linear_output);

  b.equate_outputs(p_linear_output, o_combine_output);

  return b.get_substitution();
}

//! [SubstitutionBuilder more concise example]
Substitution create_partition_conv2d_combine(positive_int num_dims,
                                             positive_int degree) {
  ASSERT(num_dims == 4_p);

  SubstitutionBuilder b;

  auto [p_input, o_input] = b.add_input(tensor_attribute_pattern_match_all());
  auto [p_weight, o_weight] = b.add_input(tensor_attribute_pattern_match_all());

  std::map<TensorSlotName, PatternValue> p_inputs = {
      {
          TensorSlotName::INPUT,
          p_input,
      },
      {
          TensorSlotName::FILTER,
          p_weight,
      },
  };

  OperatorAttributePattern conv2d_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::CONV2D),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
  }};

  std::string conv2d_name = "conv2d";
  PatternValue p_conv2d_output = insert_single_output_pattern(
      b,
      conv2d_pattern,
      p_inputs,
      /*output_pattern=*/tensor_attr_pattern_require_num_dims(num_dims),
      conv2d_name);

  OutputGraphExprValue o_partition_input_output =
      insert_partition(b, degree, ff_dim_t{0_n}, o_input);

  OutputGraphExprValue o_replicate_weights_output =
      insert_replicate(b, degree, o_weight);

  std::map<TensorSlotName, OutputGraphExprValue> o_conv2d_inputs = {
      {
          TensorSlotName::INPUT,
          o_partition_input_output,
      },
      {TensorSlotName::FILTER, o_replicate_weights_output},
  };

  OutputOperatorAttrsAssignment conv2d_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named(conv2d_name),
      {},
  };
  OutputGraphExprValue o_conv2d_output =
      insert_single_output_op(b, conv2d_expr, o_conv2d_inputs);

  OutputGraphExprValue o_combine_output =
      insert_combine(b, degree, ff_dim_t{0_n}, o_conv2d_output);

  b.equate_outputs(p_conv2d_output, o_combine_output);

  return b.get_substitution();
}
//! [SubstitutionBuilder more concise example]

Substitution create_partition_attention_combine(positive_int num_heads,
                                                positive_int degree) {

  SubstitutionBuilder b;

  auto [p_query_input, o_query_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_key_input, o_key_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_value_input, o_value_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_weights, o_weights] =
      b.add_input(tensor_attribute_pattern_match_all());
  std::map<TensorSlotName, PatternValue> p_inputs = {
      {
          TensorSlotName::QUERY,
          p_query_input,
      },
      {
          TensorSlotName::KEY,
          p_key_input,
      },
      {
          TensorSlotName::VALUE,
          p_value_input,
      },
      {
          TensorSlotName::WEIGHT,
          p_weights,
      },
  };

  OperatorAttributePattern attention_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::MULTIHEAD_ATTENTION),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
      op_attr_key_divisible_by(OperatorAttributeKey::NUM_HEADS, num_heads),
  }};

  std::string attention_name = "attention";
  PatternValue p_attention_output = insert_single_output_pattern(
      b,
      attention_pattern,
      p_inputs,
      /*output_pattern=*/tensor_attr_pattern_require_num_dims(3_p),
      attention_name);

  OutputGraphExprValue o_partition_query_input_output =
      insert_partition(b, degree, ff_dim_t{0_n}, o_query_input);

  OutputGraphExprValue o_partition_key_input_output =
      insert_partition(b, degree, ff_dim_t{0_n}, o_key_input);

  OutputGraphExprValue o_partition_value_input_output =
      insert_partition(b, degree, ff_dim_t{0_n}, o_value_input);

  OutputGraphExprValue o_replicate_weight_output =
      insert_replicate(b, degree, o_weights);

  std::map<TensorSlotName, OutputGraphExprValue> o_attention_inputs = {
      {
          TensorSlotName::QUERY,
          o_partition_query_input_output,
      },
      {
          TensorSlotName::KEY,
          o_partition_key_input_output,
      },
      {
          TensorSlotName::VALUE,
          o_partition_value_input_output,
      },
      {
          TensorSlotName::WEIGHT,
          o_replicate_weight_output,
      },
  };

  OutputOperatorAttrsAssignment attention_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named(attention_name),
      {},
  };
  OutputGraphExprValue o_attention_output =
      insert_single_output_op(b, attention_expr, o_attention_inputs);

  OutputGraphExprValue o_combine_output =
      insert_combine(b, degree, ff_dim_t{0_n}, o_attention_output);

  b.equate_outputs(p_attention_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_replicate_attention_reduce(positive_int num_heads,
                                               positive_int degree) {

  SubstitutionBuilder b;

  auto [p_query_input, o_query_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_key_input, o_key_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_value_input, o_value_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_weights, o_weights] =
      b.add_input(tensor_attribute_pattern_match_all());

  std::map<TensorSlotName, PatternValue> p_inputs = {
      {
          TensorSlotName::QUERY,
          p_query_input,
      },
      {
          TensorSlotName::KEY,
          p_key_input,
      },
      {
          TensorSlotName::VALUE,
          p_value_input,
      },
      {
          TensorSlotName::WEIGHT,
          p_weights,
      },
  };

  OperatorAttributePattern attention_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::MULTIHEAD_ATTENTION),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
      op_attr_key_divisible_by(OperatorAttributeKey::NUM_HEADS, num_heads),
  }};

  std::string attention_name = "attention";
  PatternValue p_attention_output = insert_single_output_pattern(
      b,
      attention_pattern,
      p_inputs,
      /*output_pattern=*/tensor_attr_pattern_require_num_dims(3_p),
      attention_name);

  OutputGraphExprValue o_replicate_query_input_output =
      insert_replicate(b, degree, o_query_input);

  OutputGraphExprValue o_replicate_key_input_output =
      insert_replicate(b, degree, o_key_input);

  OutputGraphExprValue o_replicate_value_input_output =
      insert_replicate(b, degree, o_value_input);

  OutputGraphExprValue o_partition_weight_output =
      insert_partition(b, degree, ff_dim_t{1_n}, o_weights);

  std::map<TensorSlotName, OutputGraphExprValue> o_attention_inputs = {
      {
          TensorSlotName::QUERY,
          o_replicate_query_input_output,
      },
      {
          TensorSlotName::KEY,
          o_replicate_key_input_output,
      },
      {
          TensorSlotName::VALUE,
          o_replicate_value_input_output,
      },
      {
          TensorSlotName::WEIGHT,
          o_partition_weight_output,
      },
  };

  OutputOperatorAttrsAssignment attention_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named(attention_name),
      {},
  };
  OutputGraphExprValue o_attention_output =
      insert_single_output_op(b, attention_expr, o_attention_inputs);

  OutputGraphExprValue o_reduce_output =
      insert_reduce(b, degree, o_attention_output);

  b.equate_outputs(p_attention_output, o_reduce_output);

  return b.get_substitution();
}

Substitution create_partition_softmax_combine(ff_dim_t softmax_dim,
                                              ff_dim_t partition_dim,
                                              positive_int degree) {
  ASSERT(partition_dim != softmax_dim);

  SubstitutionBuilder b;

  auto [p_input, o_input] = b.add_input(tensor_attribute_pattern_match_all());
  std::map<TensorSlotName, PatternValue> p_inputs = {
      {
          TensorSlotName::INPUT,
          p_input,
      },
  };

  OperatorAttributePattern softmax_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::SOFTMAX),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
      op_attr_key_divisible_by(OperatorAttributeKey::SOFTMAX_DIM,
                               positive_int{softmax_dim.value}),
  }};

  std::string softmax_name = "softmax";
  PatternValue p_softmax_output = insert_single_output_pattern(
      b,
      softmax_pattern,
      p_inputs,
      /*output_pattern=*/tensor_attribute_pattern_match_all(),
      softmax_name);

  OutputGraphExprValue o_partition_input_output =
      insert_partition(b, degree, partition_dim, o_input);

  std::map<TensorSlotName, OutputGraphExprValue> o_softmax_inputs = {
      {
          TensorSlotName::INPUT,
          o_partition_input_output,
      },
  };

  OutputOperatorAttrsAssignment softmax_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named(softmax_name),
      {},
  };
  OutputGraphExprValue o_softmax_output =
      insert_single_output_op(b, softmax_expr, o_softmax_inputs);

  OutputGraphExprValue o_combine_output =
      insert_combine(b, degree, partition_dim, o_softmax_output);

  b.equate_outputs(p_softmax_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_partition_add_combine(ff_dim_t parallel_dim,
                                          positive_int degree) {
  SubstitutionBuilder b;

  auto [p_input1, o_input1] = b.add_input(tensor_attribute_pattern_match_all());
  auto [p_input2, o_input2] = b.add_input(tensor_attribute_pattern_match_all());

  std::map<TensorSlotName, PatternValue> p_inputs = {
      {
          TensorSlotName::LHS_INPUT,
          p_input1,
      },
      {
          TensorSlotName::RHS_INPUT,
          p_input2,
      },
  };

  OperatorAttributePattern add_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::EW_ADD),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
  }};

  std::string add_name = "add";
  PatternValue p_add_output = insert_single_output_pattern(
      b,
      add_pattern,
      p_inputs,
      /*output_pattern=*/tensor_attribute_pattern_match_all(),
      add_name);

  OutputGraphExprValue o_partition_input1_output =
      insert_partition(b, degree, parallel_dim, o_input1);
  OutputGraphExprValue o_partition_input2_output =
      insert_partition(b, degree, parallel_dim, o_input2);

  std::map<TensorSlotName, OutputGraphExprValue> o_add_inputs = {
      {
          TensorSlotName::LHS_INPUT,
          o_partition_input1_output,
      },
      {
          TensorSlotName::RHS_INPUT,
          o_partition_input2_output,
      },
  };

  OutputOperatorAttrsAssignment add_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named(add_name),
      {},
  };
  OutputGraphExprValue o_add_output =
      insert_single_output_op(b, add_expr, o_add_inputs);

  OutputGraphExprValue o_combine_output =
      insert_combine(b, degree, parallel_dim, o_add_output);

  b.equate_outputs(p_add_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_partition_relu_combine(ff_dim_t parallel_dim,
                                           positive_int degree) {
  SubstitutionBuilder b;

  auto [p_input, o_input] = b.add_input(tensor_attribute_pattern_match_all());

  OperatorAttributePattern relu_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::RELU),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
  }};

  std::string relu_name = "relu";
  PatternValue p_relu_output = insert_single_output_pattern(
      b,
      relu_pattern,
      {{TensorSlotName::INPUT, p_input}},
      /*output_pattern=*/tensor_attribute_pattern_match_all(),
      relu_name);

  OutputGraphExprValue o_partition_input_output =
      insert_partition(b, degree, parallel_dim, o_input);

  OutputOperatorAttrsAssignment relu_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named(relu_name),
      {},
  };
  OutputGraphExprValue o_relu_output = insert_single_output_op(
      b, relu_expr, {{TensorSlotName::INPUT, o_partition_input_output}});

  OutputGraphExprValue o_combine_output =
      insert_combine(b, degree, parallel_dim, o_relu_output);

  b.equate_outputs(p_relu_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_fuse_linear_activation(Activation activation) {
  SubstitutionBuilder b;

  auto [p_input, o_input] =
      b.add_input(tensor_attribute_pattern_match_all(), "input");
  auto [p_weight, o_weight] =
      b.add_input(tensor_attribute_pattern_match_all(), "weight");

  OperatorAttributePattern mm_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::LINEAR),
      op_attr_key_equals(
          OperatorAttributeKey::ACTIVATION,
          OperatorAttributeValue{std::optional<Activation>{std::nullopt}}),
  }};

  std::string mm_name = "mm";
  PatternValue p_mm_output = insert_single_output_pattern(
      b,
      mm_pattern,
      /*inputs=*/
      {
          {
              TensorSlotName::INPUT,
              p_input,
          },
          {
              TensorSlotName::WEIGHT,
              p_weight,
          },
      },
      /*output_pattern=*/tensor_attribute_pattern_match_all(),
      mm_name);

  OperatorAttributePattern activation_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(op_type_for_activation(activation)),
  }};

  std::string activation_name = "activation";
  PatternValue p_activation_output = insert_single_output_pattern(
      b,
      activation_pattern,
      /*inputs=*/
      {
          {
              TensorSlotName::INPUT,
              p_mm_output,
          },
      },
      /*output_pattern=*/tensor_attribute_pattern_match_all(),
      activation_name);

  OutputOperatorAttrsAssignment fused_node_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named(mm_name),
      {
          set_attr_to_constant(OperatorAttributeKey::ACTIVATION,
                               OperatorAttributeValue{activation}),
      }};

  OutputGraphExprValue o_fused_node_output =
      insert_single_output_op(b,
                              fused_node_expr,
                              /*inputs=*/
                              {
                                  {
                                      TensorSlotName::INPUT,
                                      o_input,
                                  },
                                  {
                                      TensorSlotName::WEIGHT,
                                      o_weight,
                                  },
                              });

  b.equate_outputs(p_activation_output, o_fused_node_output);

  return b.get_substitution();
}

Substitution create_fuse_batch_norm_activation(Activation activation) {
  SubstitutionBuilder b;

  auto [p_input, o_input] =
      b.add_input(tensor_attribute_pattern_match_all(), "input");
  auto [p_gamma, o_gamma] =
      b.add_input(tensor_attribute_pattern_match_all(), "gamma");
  auto [p_beta, o_beta] =
      b.add_input(tensor_attribute_pattern_match_all(), "beta");

  OperatorAttributePattern bn_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::BATCHNORM),
      op_attr_key_equals(
          OperatorAttributeKey::ACTIVATION,
          OperatorAttributeValue{std::optional<Activation>{std::nullopt}}),
  }};

  std::string bn_name = "bn";
  PatternValue p_bn_output = insert_single_output_pattern(
      b,
      bn_pattern,
      /*inputs=*/
      {
          {
              TensorSlotName::INPUT,
              p_input,
          },
          {
              TensorSlotName::GAMMA,
              p_gamma,
          },
          {
              TensorSlotName::BETA,
              p_beta,
          },
      },
      /*output_pattern=*/tensor_attribute_pattern_match_all(),
      bn_name);

  OperatorAttributePattern activation_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(op_type_for_activation(activation)),
  }};

  std::string activation_name = "activation";
  PatternValue p_activation_output = insert_single_output_pattern(
      b,
      activation_pattern,
      /*inputs=*/
      {
          {
              TensorSlotName::INPUT,
              p_bn_output,
          },
      },
      /*output_pattern=*/tensor_attribute_pattern_match_all(),
      activation_name);

  OutputOperatorAttrsAssignment fused_node_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named(bn_name),
      {
          set_attr_to_constant(OperatorAttributeKey::ACTIVATION,
                               OperatorAttributeValue{activation}),
      }};

  OutputGraphExprValue o_fused_node_output =
      insert_single_output_op(b,
                              fused_node_expr,
                              /*inputs=*/
                              {
                                  {
                                      TensorSlotName::INPUT,
                                      o_input,
                                  },
                                  {
                                      TensorSlotName::GAMMA,
                                      o_gamma,
                                  },
                                  {
                                      TensorSlotName::BETA,
                                      o_beta,
                                  },
                              });

  b.equate_outputs(p_activation_output, o_fused_node_output);

  return b.get_substitution();
}

} // namespace FlexFlow
