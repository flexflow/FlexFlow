#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUBSTITUTION_INTERNAL_PERFORM_SHAPE_INFERENCE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUBSTITUTION_INTERNAL_PERFORM_SHAPE_INFERENCE_H

#include "pcg/parallel_computation_graph/parallel_layer_attrs.dtg.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"

namespace FlexFlow {

/**
 * @brief Takes a SubParallelComputationGraph but without ParallelTensorShape
 * annotations on its OpenDataflowValue%s and uses shape inference to fill them
 * in.
 *
 * @details The OutputGraphExpr of a Substitution only computes
 * PCGOperatorAttr%s, not ParallelTensorShape%s, under the theory that shapes
 * can be inferred by parallel shape inference. The responsibility of this
 * function is to traverse the result of evaluating the OutputGraphExpr
 * (resulting from evaluate_substitution_output)
 * and annotate each of the OpenDataflowValue%s with the inferred shape.
 *
 * Exists only to enable apply_substitution(SubParallelComputationGraph const &,
 * Substitution const &, PCGPatternMatch const &)
 */
LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorShape>
    perform_shape_inference(
        LabelledOpenDataflowGraphView<ParallelLayerAttrs, std::monostate> const
            &g,
        std::unordered_map<DataflowGraphInput, ParallelTensorShape> const
            &input_shapes);

} // namespace FlexFlow

#endif
