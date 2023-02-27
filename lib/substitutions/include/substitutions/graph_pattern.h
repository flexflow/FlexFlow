#ifndef _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H

#include "utils/graph.h"

namespace FlexFlow {
namespace substitutions {

struct InputEdge {
  utils::Node dst;
  std::size_t dstIdx;
};

struct OutputEdge {
  utils::Node src;
  std::size_t srcIdx;
};

using PatternEdge = mpark::variant<
  InputEdge,
  OutputEdge,
  utils::MultiDiEdge
>;

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

struct IMultiDiGraphPattern : public utils::IMultiDiGraph {
  std::unordered_set<utils::MultiDiEdge> query_edges(utils::MultiDiEdgeQuery const &) const override final;
  virtual std::unordered_set<PatternEdge> query_edges(PatternEdgeQuery const &) const = 0;
};

bool pattern_matches(IMultiDiGraphPattern const &, 
                     utils::IMultiDiGraph const &, 
                     std::unordered_map<utils::Node, utils::Node> const &nodeAssignment, 
                     std::unordered_map<PatternEdge, utils::MultiDiEdge> const &edgeAssignment);

}
}

namespace std {

template <>
struct hash<::FlexFlow::substitutions::PatternEdge> {
  size_t operator()(::FlexFlow::substitutions::PatternEdge const &) const;
};

}

#endif
