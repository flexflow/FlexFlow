/**
 * @file nasnet.h
 *
 * @brief NASNet model computation graphs.
 */

#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_NASNET_NASNET_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_NASNET_NASNET_H

#include "models/nasnet/nasnet_a_large_config.dtg.h"
#include "pcg/computation_graph.dtg.h"

namespace FlexFlow {

/**
 * @brief Get the default NASNet-A Large configuration.
 *
 * The defaults match the timm NASNet-A Large definition: RGB 331x331 input,
 * 1000 output classes, 4032 final features, and no classifier dropout.
 */
NASNetALargeConfig get_default_nasnet_a_large_config();

/**
 * @brief Build a NASNet-A Large computation graph.
 */
ComputationGraph
    get_nasnet_a_large_computation_graph(NASNetALargeConfig const &config);

} // namespace FlexFlow

#endif
