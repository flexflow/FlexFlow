#ifndef _FLEXFLOW_SUBSTITUTION_LOADER_H
#define _FLEXFLOW_SUBSTITUTION_LOADER_H

#include "substitution-generator/legacy_operator_type.dtg.h"
#include "substitution-generator/legacy_pm_parameter.dtg.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>

namespace FlexFlow {

struct Parameter {
  LegacyPMParameter key;
  int value;
};
void from_json(nlohmann::json const &j, Parameter &p);

struct Tensor {
  int opId;
  int tsId;
};
void from_json(nlohmann::json const &j, Tensor &t);

struct Operator {
  LegacyOperatorType op_type;
  std::vector<Tensor> input;
  std::vector<Parameter> para;

  std::optional<int> at(LegacyPMParameter key) const;
};
void from_json(nlohmann::json const &j, Operator &t);

struct MapOutput {
  int dstOpId;
  int dstTsId;
  int srcOpId;
  int srcTsId;
};
void from_json(nlohmann::json const &j, MapOutput &t);

struct Rule {
  std::string name;
  std::vector<Operator> srcOp;
  std::vector<Operator> dstOp;
  std::vector<MapOutput> mappedOutput;
};
void from_json(nlohmann::json const &j, Rule &t);

struct RuleCollection {
  std::vector<Rule> rules;
};
void from_json(nlohmann::json const &j, RuleCollection &c);

RuleCollection load_rule_collection(std::istream &s);
RuleCollection load_rule_collection_from_path(std::string const &path);

} // namespace FlexFlow

#endif // _FLEXFLOW_SUBSTITUTION_LOADER_H
