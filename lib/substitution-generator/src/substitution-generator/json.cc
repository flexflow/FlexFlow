#include "substitution-generator/json.h"
#include <cassert>
#include <functional>
#include <sstream>

using json = nlohmann::json;

namespace FlexFlow {

void from_json(json const &j, Parameter &p) {
  j.at("key").get_to(p.key);
  j.at("value").get_to(p.value);
}

void from_json(json const &j, Tensor &t) {
  j.at("opId").get_to(t.opId);
  j.at("tsId").get_to(t.tsId);
}

/* std::optional<int> Operator::at(LegacyPMParameter key) const { */
/*   std::optional<int> value = std::nullopt; */
/*   for (Parameter const &p : this->para) { */
/*     if (p.key == key) { */
/*       assert(!value.has_value()); */
/*       value = p.key; */
/*     } */
/*   } */

/*   return value; */
/* } */

void from_json(json const &j, Operator &o) {
  j.at("type").get_to(o.op_type);
  j.at("input").get_to(o.input);
  j.at("para").get_to(o.para);
}

void from_json(json const &j, MapOutput &m) {
  j.at("dstOpId").get_to(m.dstOpId);
  j.at("dstTsId").get_to(m.dstTsId);
  j.at("srcOpId").get_to(m.srcOpId);
  j.at("srcTsId").get_to(m.srcTsId);
}

void from_json(json const &j, Rule &r) {
  j.at("name").get_to(r.name);
  j.at("srcOp").get_to(r.srcOp);
  j.at("dstOp").get_to(r.dstOp);
  j.at("mappedOutput").get_to(r.mappedOutput);
}

void from_json(json const &j, RuleCollection &c) {
  j.at("rule").get_to(c.rules);
}

RuleCollection load_rule_collection(std::istream &s) {
  json j;
  s >> j;
  RuleCollection rule_collection = j;
  return rule_collection;
}

RuleCollection load_rule_collection_from_path(std::string const &path) {
  std::ifstream input(path);
  return load_rule_collection(input);
}

} // namespace FlexFlow
