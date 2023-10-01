#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_OPEN_INTERFACES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_OPEN_INTERFACES_H

#include "standard_labelled_interfaces.h"
#include "utils/graph/open_graph_interfaces.h"
#include "utils/containers.h"


// NOTE(@wmdi): I plan to remove this since we only use `OutputLabelledOpenMultiDiGraph`
namespace FlexFlow {

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = InputLabel>
struct ILabelledOpenMultiDiGraphView
    : virtual public IOpenMultiDiGraphView,
      virtual public ILabelledMultiDiGraphView<NodeLabel, EdgeLabel> {
public:
  ILabelledOpenMultiDiGraphView() = default;
  ILabelledOpenMultiDiGraphView(ILabelledOpenMultiDiGraphView const &) = delete;
  ILabelledOpenMultiDiGraphView &operator=(ILabelledOpenMultiDiGraphView const &) = delete;

  std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &q) const final {
    return map_over_unordered_set([](OpenMultiDiEdge const &e) { return get<MultiDiEdge>(e); },
      IOpenMultiDiGraphView::query_edges(static_cast<OpenMultiDiEdgeQuery>(q)));
  }

  virtual InputLabel const &at(InputMultiDiEdge const &e) const = 0;
  virtual OutputLabel const &at(OutputMultiDiEdge const &e) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(
    ILabelledOpenMultiDiGraphView<int, int, int, int>);

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = InputLabel>
struct ILabelledOpenMultiDiGraph
    : virtual public ILabelledOpenMultiDiGraphView<NodeLabel,
                                           EdgeLabel,
                                           InputLabel,
                                           OutputLabel> {
public:
  virtual ILabelledOpenMultiDiGraph *clone() const = 0;

  virtual Node add_node(NodeLabel cosnt &label) = 0;

  virtual void add_edge(MultiDiEdge const &e, Edgelabel const &label) = 0;
  virtual void add_edge(InputMultiDiEdge const &e, InputLabel const &label) = 0;
  virtual void add_edge(OutputMultiDiEdge const &e,
                        OutputLabel const &label) = 0;

  virtual InputLabel &at(InputMultiDiEdge const &e) = 0;
  virtual OutputLabel &at(OutputMultiDiEdge const &e) = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledOpenMultiDiGraph<int, int, int, int>);

} // namespace FlexFlow

#endif
