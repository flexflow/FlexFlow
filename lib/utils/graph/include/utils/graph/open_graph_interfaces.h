#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_OPEN_GRAPH_INTERFACES
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_OPEN_GRAPH_INTERFACES

#include "multidigraph.h"
#include "utils/exception.h"
#include "utils/graph/multidiedge.h"
#include "utils/graph/multidigraph_interfaces.h"
#include "utils/strong_typedef.h"
#include "utils/type_traits.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct IOpenMultiDiGraphView : virtual public IMultiDiGraphView {
  virtual std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOpenMultiDiGraphView);

struct IDownwardOpenMultiDiGraphView : virtual public IOpenMultiDiGraphView {
  virtual std::unordered_set<DownwardOpenMultiDiEdge>
      query_edges(DownwardOpenMultiDiEdgeQuery const &) const = 0;

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const final {
    return widen<OpenMultiDiEdge>(
        this->query_edges(DownwardOpenMultiDiEdgeQuery{q.output_edge_query,
                                                       q.standard_edge_query}));
  }
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDownwardOpenMultiDiGraphView);

struct IUpwardOpenMultiDiGraphView : virtual public IOpenMultiDiGraphView {
  virtual std::unordered_set<UpwardOpenMultiDiEdge>
      query_edges(UpwardOpenMultiDiEdgeQuery const &) const = 0;

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const final {
    return widen<OpenMultiDiEdge>(this->query_edges(
        UpwardOpenMultiDiEdgeQuery{q.input_edge_query, q.standard_edge_query}));
  }
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IUpwardOpenMultiDiGraphView);

struct IOpenMultiDiGraph : virtual public IOpenMultiDiGraphView {
  virtual void add_node(Node const &) = 0;
  virtual void add_edge(OpenMultiDiEdge const &) = 0;
  virtual void remove_edge(OpenMultiDiEdge const &) = 0;
  virtual IOpenMultiDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOpenMultiDiGraph);

struct IUpwardOpenMultiDiGraph : virtual public IUpwardOpenMultiDiGraphView {
  virtual void add_edge(UpwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(UpwardOpenMultiDiEdge const &) = 0;
  virtual IUpwardOpenMultiDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IUpwardOpenMultiDiGraph);

struct IDownwardOpenMultiDiGraph : virtual public IDownwardOpenMultiDiGraphView {
  virtual void add_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual IDownwardOpenMultiDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDownwardOpenMultiDiGraph);

} // namespace FlexFlow

#endif
