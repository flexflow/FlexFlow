#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_INCEPTION_V3
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_INCEPTION_V3

#include "models/inception_v3/inception_v3_config.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

InceptionV3Config get_default_inception_v3_config();

ComputationGraph
    get_inception_v3_computation_graph(InceptionV3Config const &config);

} // namespace FlexFlow

#endif
