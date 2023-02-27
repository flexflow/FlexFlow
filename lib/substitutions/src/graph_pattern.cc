#include "substitutions/graph_pattern.h"

namespace FlexFlow {
namespace substitutions {

std::unordered_set<utils::MultiDiEdge> IMultiDiGraphPattern::query_edges(utils::MultiDiEdgeQuery const &query) const {
  PatternEdgeQuery pattern_query { InputEdgeQuery::none(), query, OutputEdgeQuery::none() };
  std::unordered_set<utils::MultiDiEdge>  result;
  for (PatternEdge const &e : this->query_edges(pattern_query)) {
    result.insert(mpark::get<utils::MultiDiEdge>(e));
  }
  return result;
}

}
}
