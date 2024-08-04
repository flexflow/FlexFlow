#include "substitution-generator/legacy_rules.h"
#include "doctest/doctest.h"

using namespace FlexFlow;
using nlohmann::json;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("operator json deserialization") {
    json j = {
        {"_t", "Operator"},
        {"input",
         std::vector<json>{{{"_t", "Tensor"}, {"opId", -2}, {"tsId", 0}},
                           {{"_t", "Tensor"}, {"opId", -3}, {"tsId", 0}}}},
        {"para", std::vector<json>{}},
        {"type", "OP_EW_ADD"},
    };

    LegacyOperator o;
    from_json(j, o);

    CHECK(o.op_type == LegacyOperatorType::EW_ADD);
    CHECK(o.input.size() == 2);
    CHECK(o.input[0].opId == -2);
    CHECK(o.input[0].tsId == 0);
    CHECK(o.input[1].opId == -3);
    CHECK(o.input[1].tsId == 0);
    CHECK(o.para.size() == 0);
  }

  TEST_CASE("deserialize full file") {
    LegacyRuleCollection collection =
        load_rule_collection_from_path("graph_subst_3_v2.json");
    CHECK(collection.rules.size() == 640);
  }
}
