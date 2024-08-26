#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_EXPR_TO_RESULT_SUB_PCG_MAPPING_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_EXPR_TO_RESULT_SUB_PCG_MAPPING_H

#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "substitutions/output_expr_to_result_sub_pcg_mapping.dtg.h"
#include "substitutions/output_graph/output_graph_expr.dtg.h"
#include "substitutions/output_graph/output_graph_expr_node_output.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"

namespace FlexFlow {

bidict<parallel_tensor_guid_t, OutputGraphExprNodeOutput>
    get_output_graph_expr_output_mapping(
        OutputExprToResultSubPCGMapping const &,
        OutputGraphExpr const &,
        SubParallelComputationGraph const &);

} // namespace FlexFlow

#endif
