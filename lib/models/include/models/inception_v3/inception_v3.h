#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_INCEPTION_V3
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_INCEPTION_V3

#include "models/inception_v3/inception_v3_config.dtg.h"
#include "pcg/computation_graph.dtg.h"

namespace FlexFlow {

/**
 * @brief Get the default training config from https://arxiv.org/abs/1512.00567.
 */
InceptionV3Config get_default_inception_v3_training_config();

/**
 * @brief Get a computation graph for Inception-v3 as described in
 * https://arxiv.org/abs/1512.00567.
 */
ComputationGraph
    get_inception_v3_computation_graph(InceptionV3Config const &config);

} // namespace FlexFlow

#endif
