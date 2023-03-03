#ifndef _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H

#include "utils/graph.h"

namespace FlexFlow {
namespace substitutions {


struct MultiDiGraphPatternSubgraph : public IMultiDiGraphPatternView {
  std::unordered_set<PatternEdge> query_edges(PatternEdgeQuery const &) const override; 
  std::unordered_set<utils::Node> query_nodes(utils::NodeQuery const &) const override;
};

struct DiGraphPatternMatch {
  utils::bidict<utils::Node, utils::Node> nodeAssignment;
  utils::bidict<PatternEdge, utils::MultiDiEdge> edgeAssignment;
};


using GraphSplit = std::pair<std::unordered_set<utils::Node>, std::unordered_set<utils::Node>>;

struct MatchSplit {
  DiGraphPatternMatch prefix_submatch;
  DiGraphPatternMatch postfix_submatch;
};

GraphSplit split_pattern(IMultiDiGraphPatternView const &pattern);

struct ViewPatternAsMultiDiGraph : public utils::IMultiDiGraphView { 
public:
  ViewPatternAsMultiDiGraph() = delete;
  explicit ViewPatternAsMultiDiGraph(IMultiDiGraphPatternView const &);

  std::unordered_set<utils::MultiDiEdge> query_edges(utils::MultiDiEdgeQuery const &) const override;
  std::unordered_set<utils::Node> query_nodes(utils::NodeQuery const &) const override;
private:
  IMultiDiGraphPatternView const &pattern;
};

struct MultiDiGraphPatternSubgraphView : public IMultiDiGraphPatternView {
public:
  MultiDiGraphPatternSubgraphView() = delete;
  explicit MultiDiGraphPatternSubgraphView(IMultiDiGraphPatternView const &, std::unordered_set<utils::Node> const &);

  std::unordered_set<PatternEdge> query_edges(PatternEdgeQuery const &) const override;
  std::unordered_set<utils::Node> query_nodes(utils::NodeQuery const &) const override;
private:
  IMultiDiGraphPatternView const &pattern;
};

std::unique_ptr<utils::IMultiDiGraphView> unsafe_view_as_multidigraph(IMultiDiGraphPatternView const &);
std::unique_ptr<IMultiDiGraphPatternView> unsafe_view_as_subgraph(IMultiDiGraphPatternView const &, std::unordered_set<utils::Node> const &);
std::vector<utils::Node> get_topological_ordering(IMultiDiGraphPatternView const &);


std::unordered_set<PatternEdge> get_edges(IMultiDiGraphPatternView const &);

bool pattern_matches(IMultiDiGraphPattern const &, 
                     utils::IMultiDiGraph const &, 
                     DiGraphPatternMatch const &);
bool is_singleton_pattern(IMultiDiGraphPatternView const &);

}
}

namespace std {

template <>
struct hash<::FlexFlow::substitutions::DiGraphPatternMatch> {
  size_t operator()(::FlexFlow::substitutions::DiGraphPatternMatch const &) const;
};

}

#endif
