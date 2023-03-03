#include "flexflow/substitution.h"
#include "flexflow/substitution_loader.h"
#include "gtest/gtest.h"

namespace sl = FlexFlow::substitution_loader;
// using namespace FlexFlow::substitution_loader;
using json = nlohmann::json;
using FlexFlow::PCG::create_xfer;
using FlexFlow::PCG::create_xfers;
using FlexFlow::PCG::GraphXfer;

TEST(substitution_loader, basic) {
  // Yes, I know this substitution is not correct. It's just for testing.

  sl::Rule example_rule;

  example_rule.name = "test_rule";

  sl::Tensor input_tensor1;
  input_tensor1.opId = -1;
  input_tensor1.tsId = 0;

  sl::Tensor input_tensor2;
  input_tensor2.opId = -2;
  input_tensor2.tsId = 0;

  sl::Operator srcOp1;
  srcOp1.op_type = OP_EW_ADD;
  srcOp1.input = {input_tensor1, input_tensor2};
  srcOp1.para = {};

  sl::Tensor srcOp1Output;
  srcOp1Output.opId = 0;
  srcOp1Output.tsId = 0;

  sl::Parameter activation_constraint;
  activation_constraint.key = PM_ACTI;
  activation_constraint.value = AC_MODE_NONE;

  sl::Operator srcOp2;
  srcOp2.op_type = OP_LINEAR;
  srcOp2.input = {srcOp1Output};
  srcOp2.para = {activation_constraint};

  sl::Operator dstOp1;
  dstOp1.op_type = OP_LINEAR;
  dstOp1.input = {input_tensor1};
  dstOp1.para = {activation_constraint};

  sl::Tensor dstOp1Output;
  dstOp1Output.opId = 0;
  dstOp1Output.tsId = 0;

  sl::Operator dstOp2;
  dstOp2.op_type = OP_LINEAR;
  dstOp2.input = {input_tensor2};
  dstOp2.para = {activation_constraint};

  sl::Tensor dstOp2Output;
  dstOp2Output.opId = 1;
  dstOp2Output.tsId = 0;

  sl::Operator dstOp3;
  dstOp3.op_type = OP_EW_ADD;
  dstOp3.input = {dstOp1Output, dstOp2Output};
  dstOp3.para = {};

  sl::MapOutput map_output;
  map_output.srcOpId = 1;
  map_output.srcTsId = 0;
  map_output.dstOpId = 2;
  map_output.dstTsId = 0;

  example_rule.srcOp = {srcOp1, srcOp2};
  example_rule.dstOp = {dstOp1, dstOp2, dstOp3};
  example_rule.mappedOutput = {map_output};

  GraphXfer *xfer = new GraphXfer(nullptr);
  create_xfer(*xfer, example_rule, 2);

  EXPECT_EQ(xfer->name, "test_rule");

  EXPECT_EQ(xfer->srcOps.size(), 2);
  EXPECT_EQ(xfer->srcOps[0]->type, OP_EW_ADD);
  EXPECT_EQ(xfer->srcOps[1]->type, OP_LINEAR);
  EXPECT_EQ(xfer->srcOps[0]->inputs.size(), 2);
  EXPECT_NE(xfer->srcOps[0]->inputs[0], xfer->srcOps[0]->inputs[1]);
  EXPECT_EQ(xfer->srcOps[0]->outputs.size(), 1);
  EXPECT_EQ(xfer->srcOps[1]->inputs.size(), 1);
  EXPECT_EQ(xfer->srcOps[0]->outputs[0], xfer->srcOps[1]->inputs[0]);
  EXPECT_EQ(xfer->srcOps[1]->outputs.size(), 1);

  EXPECT_EQ(xfer->dstOps.size(), 3);
  EXPECT_EQ(xfer->dstOps[0]->type, OP_LINEAR);
  EXPECT_EQ(xfer->dstOps[1]->type, OP_LINEAR);
  EXPECT_EQ(xfer->dstOps[2]->type, OP_EW_ADD);
  EXPECT_EQ(xfer->dstOps[0]->inputs.size(), 1);
  EXPECT_EQ(xfer->dstOps[0]->outputs.size(), 1);
  EXPECT_EQ(xfer->dstOps[0]->inputs[0], xfer->srcOps[0]->inputs[0]);
  EXPECT_EQ(xfer->dstOps[1]->inputs.size(), 1);
  EXPECT_EQ(xfer->dstOps[1]->outputs.size(), 1);
  EXPECT_EQ(xfer->dstOps[1]->inputs[0], xfer->srcOps[0]->inputs[1]);
  EXPECT_EQ(xfer->dstOps[2]->inputs.size(), 2);
  EXPECT_EQ(xfer->dstOps[2]->inputs[0], xfer->dstOps[0]->outputs[0]);
  EXPECT_EQ(xfer->dstOps[2]->inputs[1], xfer->dstOps[1]->outputs[0]);
  EXPECT_NE(xfer->dstOps[2]->inputs[0], xfer->dstOps[2]->inputs[1]);
  EXPECT_EQ(xfer->dstOps[2]->outputs.size(), 1);

  EXPECT_EQ(xfer->mappedOutputs.size(), 1);
  EXPECT_NE(xfer->srcOps[1]->outputs[0], xfer->dstOps[2]->outputs[0]);
  EXPECT_EQ(xfer->mappedOutputs.at(xfer->srcOps[1]->outputs[0]),
            xfer->dstOps[2]->outputs[0]);
}

TEST(substitution_loader, operator_deserialization) {
  json j = {
      {"_t", "Operator"},
      {"input",
       std::vector<json>{{{"_t", "Tensor"}, {"opId", -2}, {"tsId", 0}},
                         {{"_t", "Tensor"}, {"opId", -3}, {"tsId", 0}}}},
      {"para", std::vector<json>{}},
      {"type", "OP_EW_ADD"},
  };

  sl::Operator o;
  from_json(j, o);

  EXPECT_EQ(o.op_type, OP_EW_ADD);
  EXPECT_EQ(o.input.size(), 2);
  EXPECT_EQ(o.input[0].opId, -2);
  EXPECT_EQ(o.input[0].tsId, 0);
  EXPECT_EQ(o.input[1].opId, -3);
  EXPECT_EQ(o.input[1].tsId, 0);
  EXPECT_EQ(o.para.size(), 0);
}

// TEST(substitution_loader, load_full_file) {
//   sl::RuleCollection collection =
//       sl::load_rule_collection_from_path("tests/unit/graph_subst_3_v2.json");
//   EXPECT_EQ(collection.rules.size(), 640);

//   std::vector<GraphXfer *> xfers = create_xfers(nullptr, collection, 2);
//   EXPECT_EQ(xfers.size(), 640);
// }
