#ifndef _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H

#include "utils/graph.h"

namespace FlexFlow {
namespace substitutions {


struct MultiDiGraphPatternSubgraph : public IOpenMultiDiGraph {
  std::unordered_set<OpenMultiDiEdge> query_edges(OpenMultiDiEdgeQuery const &) const override; 
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
};

struct DiGraphPatternMatch {
  bidict<Node, Node> nodeAssignment;
  bidict<OpenMultiDiEdge, MultiDiEdge> edgeAssignment;
};


using GraphSplit = std::pair<std::unordered_set<Node>, std::unordered_set<Node>>;

struct MatchSplit {
  DiGraphPatternMatch prefix_submatch;
  DiGraphPatternMatch postfix_submatch;
};

GraphSplit split_pattern(IOpenMultiDiGraph const &pattern);

struct ViewPatternAsMultiDiGraph : public IMultiDiGraphView { 
public:
  ViewPatternAsMultiDiGraph() = delete;
  explicit ViewPatternAsMultiDiGraph(IOpenMultiDiGraph const &);

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  IOpenMultiDiGraphView const &pattern;
};

struct MultiDiGraphPatternSubgraphView : public IOpenMultiDiGraphView {
public:
  MultiDiGraphPatternSubgraphView() = delete;
  explicit MultiDiGraphPatternSubgraphView(IOpenMultiDiGraphView const &, std::unordered_set<Node> const &);

  std::unordered_set<OpenMultiDiEdge> query_edges(OpenMultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  IOpenMultiDiGraphView const &pattern;
};

std::unique_ptr<IMultiDiGraphView> unsafe_view_as_multidigraph(IOpenMultiDiGraphView const &);
std::unique_ptr<IOpenMultiDiGraphView> unsafe_view_as_subgraph(IOpenMultiDiGraphView const &, std::unordered_set<Node> const &);
std::vector<Node> get_topological_ordering(IOpenMultiDiGraphView const &);


std::unordered_set<OpenMultiDiEdge> get_edges(IOpenMultiDiGraphView const &);

bool pattern_matches(IOpenMultiDiGraphView const &, 
                     IMultiDiGraph const &, 
                     DiGraphPatternMatch const &);
bool is_singleton_pattern(IOpenMultiDiGraphView const &);

}
}

namespace std {

template <>
struct hash<::FlexFlow::substitutions::DiGraphPatternMatch> {
  size_t operator()(::FlexFlow::substitutions::DiGraphPatternMatch const &) const;
};

}

#endif
