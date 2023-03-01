#ifndef _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H

#include "utils/graph.h"

namespace FlexFlow {
namespace substitutions {

struct InputEdge {
  utils::Node dst;
  std::size_t dstIdx;
};
bool operator==(InputEdge const &, InputEdge const &);

struct OutputEdge {
  utils::Node src;
  std::size_t srcIdx;
};
bool operator==(OutputEdge const &, OutputEdge const &);

using PatternEdge = mpark::variant<
  InputEdge,
  OutputEdge,
  utils::MultiDiEdge
>;

bool is_input_edge(PatternEdge const &);
bool is_output_edge(PatternEdge const &);
bool is_standard_edge(PatternEdge const &);

struct OutputEdgeQuery {
  tl::optional<std::unordered_set<utils::Node>> srcs = tl::nullopt;
  tl::optional<std::unordered_set<std::size_t>> srcIdxs = tl::nullopt;

  static OutputEdgeQuery all();
  static OutputEdgeQuery none();
};

struct InputEdgeQuery {
  tl::optional<std::unordered_set<utils::Node>> dsts = tl::nullopt;
  tl::optional<std::unordered_set<std::size_t>> dstIdxs = tl::nullopt;

  static InputEdgeQuery all();
  static InputEdgeQuery none();
};

struct PatternEdgeQuery {
  InputEdgeQuery input_edge_query;
  utils::MultiDiEdgeQuery standard_edge_query;
  OutputEdgeQuery output_edge_query;
};

}
}

namespace std {

template <>
struct hash<::FlexFlow::substitutions::PatternEdge> {
  size_t operator()(::FlexFlow::substitutions::PatternEdge const &) const;
};

template <>
struct hash<::FlexFlow::substitutions::OutputEdge> {
  size_t operator()(::FlexFlow::substitutions::OutputEdge const &) const;
};

template <>
struct hash<::FlexFlow::substitutions::InputEdge> {
  size_t operator()(::FlexFlow::substitutions::InputEdge const &) const;
};

}

namespace FlexFlow {
namespace substitutions {

struct IMultiDiGraphPatternView : public utils::IGraphView {
  virtual std::unordered_set<PatternEdge> query_edges(PatternEdgeQuery const &) const = 0;
};

struct IMultiDiGraphPattern : public IMultiDiGraphPatternView, public utils::IGraph {
  virtual void add_edge(PatternEdge const &) = 0;
  virtual void remove_edge(PatternEdge const &) = 0;
};

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
