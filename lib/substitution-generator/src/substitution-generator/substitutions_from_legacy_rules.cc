#include "substitution-generator/substitutions_from_legacy_rules.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"

namespace FlexFlow {

OperatorAttributeKey operator_attribute_key_from_legacy(LegacyPMParameter) {
  NOT_IMPLEMENTED();
}

OperatorType operator_type_from_legacy(LegacyOperatorType) {
  NOT_IMPLEMENTED();
}

OperatorAttributeConstraint operator_attribute_constraint_from_legacy(LegacyParameter const &) {
  NOT_IMPLEMENTED();
}

std::unordered_map<std::string, Substitution> substitutions_from_legacy_rules(LegacyRuleCollection const &legacy) {
}

Substitution substitution_from_legacy_rule(LegacyRule const &legacy) {
  LabelledOpenDataflowGraph<OperatorAttributePattern, TensorAttributePattern> pattern_g = 
    LabelledOpenDataflowGraph<OperatorAttributePattern, TensorAttributePattern>
    ::create<UnorderedSetLabelledOpenDataflowGraph<OperatorAttributePattern, TensorAttributePattern>>();

  for (LegacyOperator const &legacy_op : legacy.srcOp) {
    OperatorAttributePattern op_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(operator_type_from_legacy(legacy_op.op_type)),
    }};

    
  }


}

} // namespace FlexFlow
