#include "utils/graph/views.h"
#include "utils/containers.h"

namespace FlexFlow {
namespace utils {

FlippedView::FlippedView(IDiGraphView const &g)
  : g(g)
{ }

std::unordered_set<DirectedEdge> FlippedView::query_edges(DirectedEdgeQuery const &query) const {
  std::unordered_set<DirectedEdge> result = this->g.query_edges({query.dsts, query.srcs});
  return map_over_unordered_set<DirectedEdge, DirectedEdge>(flipped, result);
}

std::unordered_set<Node> FlippedView::query_nodes(NodeQuery const &query) const {
  return this->g.query_nodes(query);
}

DirectedEdge flipped(DirectedEdge const &e) {
  return {e.src, e.dst};  
}


FlippedView unsafe_view_as_flipped(IDiGraphView const &g) {
  return FlippedView(g);
}

DiSubgraphView::DiSubgraphView(IDiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes)
  : g(g), subgraph_nodes(subgraph_nodes)
{ } 

std::unordered_set<DirectedEdge> DiSubgraphView::query_edges(DirectedEdgeQuery const &query) const {
  DirectedEdgeQuery subgraph_query = {this->subgraph_nodes, this->subgraph_nodes};
  return this->g.query_edges(query_intersection(query, subgraph_query));
}

std::unordered_set<Node> DiSubgraphView::query_nodes(NodeQuery const &query) const {
  return this->g.query_nodes(query_intersection(query, {this->subgraph_nodes}));
}

MultiDiSubgraphView::MultiDiSubgraphView(IMultiDiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes)
  : g(g), subgraph_nodes(subgraph_nodes)
{ }

std::unordered_set<MultiDiEdge> MultiDiSubgraphView::query_edges(MultiDiEdgeQuery const &query) const {
  MultiDiEdgeQuery subgraph_query = MultiDiEdgeQuery::all().with_src_nodes(this->subgraph_nodes).with_dst_nodes(this->subgraph_nodes);
  return this->g.query_edges(query_intersection(query, subgraph_query));
}

DiSubgraphView unsafe_view_subgraph(IDiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes) {
  return DiSubgraphView(g, subgraph_nodes);
}

MultiDiSubgraphView unsafe_view_subgraph(IMultiDiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes) {
  return MultiDiSubgraphView(g, subgraph_nodes);
}
  
}
}
