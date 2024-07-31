#ifndef _FLEXFLOW_LIB_SUBSTITUTION_GENERATOR_INCLUDE_SUBSTITUTION_GENERATOR_SUBSTITUTIONS_FROM_LEGACY_RULES_H
#define _FLEXFLOW_LIB_SUBSTITUTION_GENERATOR_INCLUDE_SUBSTITUTION_GENERATOR_SUBSTITUTIONS_FROM_LEGACY_RULES_H

#include "substitution-generator/legacy_rules.h"
#include "substitutions/substitution.h"

namespace FlexFlow {

std::unordered_map<std::string, Substitution> substitutions_from_legacy_rules(LegacyRuleCollection const &);
Substitution substitution_from_legacy_rule(LegacyRule const &);

OperatorAttributeConstraint operator_attribute_constraint_from_legacy(LegacyParameter const &);
OperatorAttributeKey operator_attribute_key_from_legacy(LegacyPMParameter);
OperatorType operator_type_from_legacy(LegacyOperatorType);
} // namespace FlexFlow

#endif
