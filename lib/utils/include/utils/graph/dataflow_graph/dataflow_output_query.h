#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_OUTPUT_QUERY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_OUTPUT_QUERY_H

#include "utils/graph/dataflow_graph/dataflow_output.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_output_query.dtg.h"

namespace FlexFlow {

DataflowOutputQuery dataflow_output_query_all();
DataflowOutputQuery dataflow_output_query_none();
bool dataflow_output_query_includes_dataflow_output(DataflowOutputQuery const &,
                                                    DataflowOutput const &);
DataflowOutputQuery dataflow_output_query_for_output(DataflowOutput const &);
std::unordered_set<DataflowOutput> apply_dataflow_output_query(DataflowOutputQuery const &, std::unordered_set<DataflowOutput> const &);

} // namespace FlexFlow

#endif
