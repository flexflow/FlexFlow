#ifndef _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_INTERNAL_H
#define _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_INTERNAL_H

#include "utils/graph/serial_parallel/sink_settings.dtg.h"
#include "utils/graph/serial_parallel/source_settings.dtg.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/serial_parallel/intermediate_sp_decomposition_tree.dtg.h"
#include "utils/graph/serial_parallel/serialparallel.h"
#include "utils/visitable.h"
#include <variant>
#include <vector>

namespace FlexFlow {

Node find_source_node(DiGraphView const &);
Node find_sink_node(DiGraphView const &);
std::optional<Node> find_bottleneck_node(DiGraphView const &);

std::variant<IntermediateSpDecompositionTree, Node>
    sp_decomposition(DiGraphView const &g);
IntermediateSpDecompositionTree parallel_decomposition(DiGraphView const &g);

std::unordered_set<Node>
    from_source_to_sink(DiGraphView const &, Node const &src, Node const &sink);
std::unordered_set<Node>
    from_source_to_sink(DiGraphView const &g,
                        std::unordered_set<Node> const &srcs,
                        std::unordered_set<Node> const &sinks,
                        SourceSettings include_src,
                        SinkSettings include_sink);
DiGraphView source_to_sink_subgraph(DiGraphView const &g,
                                    std::unordered_set<Node> const &srcs,
                                    std::unordered_set<Node> const &sinks,
                                    SourceSettings include_src,
                                    SinkSettings include_sink);

} // namespace FlexFlow

#endif
