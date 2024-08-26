#include "substitutions/example_substitutions.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/output_graph/output_pattern_value.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "substitutions/unlabelled/pattern_value.dtg.h"

namespace FlexFlow {

// struct SubstitutionBuilder {
//   SubstitutionBuilder();
//
//   PatternValue add_input(std::string const &name,
//   std::optional<TensorAttributePattern> const & = std::nullopt);
//
//   std::vector<PatternValue> add_operator_to_pattern(std::string const &name,
//                                          OperatorAttributePattern const &,
//                                          std::vector<PatternValue> const
//                                          &inputs,
//                                          std::vector<TensorAttributePattern>
//                                          const &outputs);
//   PatternValue add_operator_to_pattern(std::string const &name,
//                             OperatorAttributePattern const &,
//                             std::vector<PatternValue> const &inputs,
//                             TensorAttributePattern const &output);
//
//   OutputPatternValue add_input_to_output(PatternValue const &);
//   std::vector<OutputPatternValue>
//   add_operator_to_output(OutputOperatorAttrsAssignment const &,
//                                             std::vector<OutputPatternValue>
//                                             const &inputs, int num_outputs);
//   OutputPatternValue add_operator_to_output(OutputOperatorAttrsAssignment
//   const &,
//                                             std::vector<OutputPatternValue>
//                                             const &inputs);
//
//   PatternNode pattern_node(std::string const &name) const;
//
//   Substitution get_substitution();
//
//   void unify_outputs(PatternValue const &, OutputPatternValue const &);
// private:
//   LabelledOpenDataflowGraph<OperatorAttributePattern, TensorAttributePattern>
//   g;
// };
//
// Substitution create_replicate_linear_combine(int num_dims,
//                                              int result_degree) {
//   SubstitutionBuilder b;
//
//   PatternValue input = b.add_input("input", TensorAttributePattern{{
//     TensorAttributeConstraint{
//       ConstraintType::EQUAL,
//       TensorAttributeExpr{
//         TensorAttributeListSize{
//           TensorAttributeKey::DIM_SIZES,
//         },
//       },
//       TensorAttributeValue{num_dims},
//     },
//   }});
//
//   PatternValue weight = b.add_input("weight");
//
//   PatternValue output = b.add_operator_to_pattern(
//     "linear",
//     OperatorAttributePattern{{
//       op_type_equals_constraint(OperatorType::LINEAR),
//     }},
//     {
//       input,
//       weight,
//     },
//     tensor_attribute_pattern_match_all()
//   );
//
//   OutputOperatorAttrsAssignment new_replicate =
//   OutputOperatorAttrsAssignment{{
//     {
//       OperatorAttributeKey::OP_TYPE,
//       OperatorAttributeValue{OperatorType::REPLICATE},
//     },
//     {
//       OperatorAttributeKey::PARALLEL_DEGREE,
//       OperatorAttributeValue{result_degree},
//     },
//   }};
//
//
//   OutputOperatorAttrsAssignment new_linear =
//   output_operator_clone_node(b.pattern_node("linear"));
//
//   OutputOperatorAttrsAssignment new_combine = OutputOperatorAttrsAssignment{{
//     {
//       OperatorAttributeKey::OP_TYPE,
//       OperatorAttributeValue{OperatorType::COMBINE},
//     },
//     {
//       OperatorAttributeKey::PARALLEL_DEGREE,
//       OperatorAttributeValue{result_degree},
//     },
//     {
//       OperatorAttributeKey::PARALLEL_DIM,
//       OperatorAttributeValue{
//         num_dims - 2
//       },
//     }
//   }};
//
//   OutputPatternValue t = b.add_input_to_output(input);
//   t = b.add_operator_to_output(new_replicate, {t});
//   t = b.add_operator_to_output(new_linear, {t,
//   b.add_input_to_output(weight)}); t = b.add_operator_to_output(new_combine,
//   {t});
//
//   b.unify_outputs(output, t);
//
//   return b.get_substitution();
//
//   // the other option is to have the user build the pattern and the output
//   first, and only then do the mapping at the end. Not sure which is better
// }
//
// Substitution create_linear_relu_merge(int num_dims) {
//   SubstitutionBuilder b;
//
//   PatternValue input = b.add_input("input", TensorAttributePattern{
//
//   });
// }

} // namespace FlexFlow
