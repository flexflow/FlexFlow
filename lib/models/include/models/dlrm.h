#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_DLRM_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_DLRM_H

#include "models/dlrm_config.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

// Helper functions to construct the DLRM model

DLRMConfig get_default_dlrm_config();

/**
 * @brief Get the DLRM computation graph.
 *
 * @param DLRMConfig The config of DLRM model.
 * @return ComputationGraph The PCG of a DLRM model.
 */
ComputationGraph get_dlrm_computation_graph(DLRMConfig const &);

} // namespace FlexFlow

#endif
