#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H

#include "models/transformer.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

ComputationGraph get_transformer_computation_graph(TransformerConfig const &);

} // namespace FlexFlow

#endif
