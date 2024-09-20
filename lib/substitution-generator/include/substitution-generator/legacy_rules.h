#ifndef _FLEXFLOW_LIB_SUBSTITUTION_GENERATOR_INCLUDE_SUBSTITUTION_GENERATOR_LEGACY_RULES_H
#define _FLEXFLOW_LIB_SUBSTITUTION_GENERATOR_INCLUDE_SUBSTITUTION_GENERATOR_LEGACY_RULES_H

#include "substitution-generator/legacy_operator_type.dtg.h"
#include "substitution-generator/legacy_pm_parameter.dtg.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>

namespace FlexFlow {

struct LegacyParameter {
  LegacyPMParameter key;
  int value;
};
void from_json(nlohmann::json const &j, LegacyParameter &p);

struct LegacyTensor {
  int opId;
  int tsId;
};
void from_json(nlohmann::json const &j, LegacyTensor &t);

struct LegacyOperator {
  LegacyOperatorType op_type;
  std::vector<LegacyTensor> input;
  std::vector<LegacyParameter> para;

  std::optional<int> at(LegacyPMParameter key) const;
};
void from_json(nlohmann::json const &j, LegacyOperator &t);

struct LegacyMapOutput {
  int dstOpId;
  int dstTsId;
  int srcOpId;
  int srcTsId;
};
void from_json(nlohmann::json const &j, LegacyMapOutput &t);

struct LegacyRule {
  std::string name;
  std::vector<LegacyOperator> srcOp;
  std::vector<LegacyOperator> dstOp;
  std::vector<LegacyMapOutput> mappedOutput;
};
void from_json(nlohmann::json const &j, LegacyRule &t);

struct LegacyRuleCollection {
  std::vector<LegacyRule> rules;
};
void from_json(nlohmann::json const &j, LegacyRuleCollection &c);

LegacyRuleCollection load_rule_collection(std::istream &s);
LegacyRuleCollection load_rule_collection_from_path(std::string const &path);

} // namespace FlexFlow

#endif // _FLEXFLOW_SUBSTITUTION_LOADER_H
