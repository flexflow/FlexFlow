#include "compiler/machine_mapping/allowed_machine_mappings.h"
#include "utils/containers/get_first.h"
#include "utils/containers/set_minus.h"
#include "utils/containers.h"
#include "utils/containers/keys.h"

namespace FlexFlow {

std::vector<std::unordered_map<Node, MachineView>>
    allowed_machine_mappings(MachineMappingContext const &context,
                             std::unordered_set<Node> const &nodes,
                             MachineSpecification const &resource) {
  if (nodes.empty()) {
    return {{}};
  }
  Node node = get_first(nodes);
  std::vector<std::unordered_map<Node, MachineView>> partial_enumeration =
      allowed_machine_mappings(context, set_minus(nodes, {node}), resource);
  std::unordered_set<MachineView> allowed_machine_views_for_node =
      context.allowed_machine_views(context.pcg.raw_graph.at(node), resource);
  std::vector<std::unordered_map<Node, MachineView>> enumeration;
  for (MachineView const &mv : allowed_machine_views_for_node) {
    for (std::unordered_map<Node, MachineView> const &partial :
         partial_enumeration) {
      enumeration.push_back(merge_maps(
          partial, std::unordered_map<Node, MachineView>{{node, mv}}));
    }
  }
  return enumeration;
}

std::vector<std::unordered_map<DataflowOutput, MachineView>>
    allowed_machine_mappings(MachineMappingContext const &context,
                             std::unordered_set<DataflowOutput> const &values,
                             MachineSpecification const &resource) {
  std::unordered_set<Node> nodes;
  for (DataflowOutput const &v : values) {
    nodes.insert(v.node);
  }

  std::vector<std::unordered_map<Node, MachineView>> node_enumeration =
      allowed_machine_mappings(context, nodes, resource);
  std::vector<std::unordered_map<DataflowOutput, MachineView>> enumeration;

  for (std::unordered_map<Node, MachineView> _node_enumeration :
       node_enumeration) {
    std::unordered_map<DataflowOutput, MachineView> _emumeration;
    for (DataflowOutput const &v : values) {
      _emumeration.emplace(v, _node_enumeration.at(v.node));
    }
    enumeration.push_back(_emumeration);
  }

  return enumeration;
}

}
