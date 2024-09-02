#include "substitution-generator/legacy_rules.h"
#include "utils/dot_file.h"
#include <cassert>
#include <iostream>

using namespace FlexFlow;

enum class NodeType {
  SRC,
  DST,
  SRC_INPUT_TENSOR,
  DST_INPUT_TENSOR,
  SRC_OUTPUT_TENSOR,
  DST_OUTPUT_TENSOR,
};

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <json-file> <rule-name>"
              << std::endl;
    return 1;
  }

  std::string json_path(argv[1]);
  std::string rule_name(argv[2]);

  LegacyRuleCollection rule_collection = load_rule_collection_from_path(json_path);

  std::optional<LegacyRule> found = std::nullopt;
  for (LegacyRule const &r : rule_collection.rules) {
    if (r.name == rule_name) {
      found = r;
      break;
    }
  }

  if (!found.has_value()) {
    std::cerr << "Could not find rule with name " << rule_name << std::endl;
    return 1;
  }

  LegacyRule r = found.value();

  using Node = std::tuple<NodeType, int, int>;

  DotFile<Node> dot(std::cout);
  size_t src_subgraph = dot.add_subgraph();
  size_t dst_subgraph = dot.add_subgraph();
  size_t src_input_subgraph = dot.add_subgraph(src_subgraph);
  size_t dst_input_subgraph = dot.add_subgraph(dst_subgraph);
  size_t src_output_subgraph = dot.add_subgraph(src_subgraph);
  size_t dst_output_subgraph = dot.add_subgraph(dst_subgraph);
  size_t src_body_subgraph = dot.add_subgraph(src_subgraph);
  size_t dst_body_subgraph = dot.add_subgraph(dst_subgraph);

  auto label_map = [](std::string const &name, Node const &n) {
    std::string color;
    switch (std::get<0>(n)) {
      case NodeType::SRC_INPUT_TENSOR:
        color = "green";
        break;
      case NodeType::DST_INPUT_TENSOR:
        color = "brown";
        break;
      case NodeType::SRC:
        color = "blue";
        break;
      case NodeType::DST:
        color = "red";
        break;
      case NodeType::SRC_OUTPUT_TENSOR:
        color = "yellow";
        break;
      case NodeType::DST_OUTPUT_TENSOR:
        color = "purple";
        break;
      default:
        assert(false);
    }
    std::map<std::string, std::string> m{{"color", color}, {"label", name}};
    return m;
  };

  for (int i = 0; i < r.srcOp.size(); i++) {
    LegacyOperator const &o = r.srcOp[i];
    Node srcOpNode = {NodeType::SRC, i, 0};
    {
      dot.add_node(srcOpNode, label_map(fmt::to_string(o.op_type), srcOpNode));
      dot.add_node_to_subgraph(srcOpNode, src_body_subgraph);
    }

    for (LegacyTensor const &t : o.input) {
      if (t.opId < 0) {
        assert(t.tsId == 0);
        Node inputOpNode = {NodeType::SRC_INPUT_TENSOR, t.opId, 0};

        {
          dot.add_node(inputOpNode, label_map("INPUT", inputOpNode));
          dot.add_node_to_subgraph(inputOpNode, src_input_subgraph);
          dot.add_edge(inputOpNode, srcOpNode);
          dot.add_edge(inputOpNode, {NodeType::DST_INPUT_TENSOR, t.opId, 0});
        }
      } else {
        dot.add_edge({NodeType::SRC, t.opId, 0}, {NodeType::SRC, i, 0});
      }
    }
  }
  for (int j = 0; j < r.dstOp.size(); j++) {
    LegacyOperator const &o = r.dstOp[j];
    Node dstOpNode = {NodeType::DST, j, 0};
    {
      dot.add_node(dstOpNode, label_map(fmt::to_string(o.op_type), dstOpNode));
      dot.add_node_to_subgraph(dstOpNode, dst_body_subgraph);
    }

    for (LegacyTensor const &t : o.input) {
      if (t.opId < 0) {
        assert(t.tsId == 0);
        Node inputOpNode = {NodeType::DST_INPUT_TENSOR, t.opId, 0};

        {
          dot.add_node(inputOpNode, label_map("INPUT", inputOpNode));
          dot.add_node_to_subgraph(inputOpNode, dst_input_subgraph);
          dot.add_edge(inputOpNode, dstOpNode);
        }
      } else {
        dot.add_edge({NodeType::DST, t.opId, 0}, {NodeType::DST, j, 0});
      }
    }
  }
  for (LegacyMapOutput const &mo : r.mappedOutput) {
    Node srcOutputNode = {NodeType::SRC_OUTPUT_TENSOR, mo.srcOpId, mo.srcTsId};
    Node dstOutputNode = {NodeType::DST_OUTPUT_TENSOR, mo.dstOpId, mo.dstTsId};
    {
      dot.add_node(srcOutputNode, label_map("OUTPUT", srcOutputNode));
      dot.add_node_to_subgraph(srcOutputNode, src_output_subgraph);
    }
    dot.add_edge({NodeType::SRC, mo.srcOpId, 0},
                 {NodeType::SRC_OUTPUT_TENSOR, mo.srcOpId, mo.srcTsId});
    {
      dot.add_node(dstOutputNode, label_map("OUTPUT", dstOutputNode));
      dot.add_node_to_subgraph(dstOutputNode, dst_output_subgraph);
    }
    dot.add_edge({NodeType::DST, mo.dstOpId, 0},
                 {NodeType::DST_OUTPUT_TENSOR, mo.dstOpId, mo.dstTsId});
    dot.add_edge({NodeType::SRC_OUTPUT_TENSOR, mo.srcOpId, mo.srcTsId},
                 {NodeType::DST_OUTPUT_TENSOR, mo.dstOpId, mo.dstTsId});
  }
  dot.close();
  std::cout << std::endl;
}
