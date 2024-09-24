#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_DLRM_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_DLRM_H

#include "models/dlrm/dlrm_config.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

// Helper functions to construct the DLRM model

DLRMConfig get_default_dlrm_config();

tensor_guid_t create_dlrm_mlp(ComputationGraphBuilder &,
                              DLRMConfig const &,
                              tensor_guid_t const &,
                              std::vector<size_t> const &,
                              int const &);

tensor_guid_t create_dlrm_emb(ComputationGraphBuilder &,
                              DLRMConfig const &,
                              tensor_guid_t const &,
                              int const &,
                              int const &);

tensor_guid_t create_dlrm_interact_features(ComputationGraphBuilder &,
                                            DLRMConfig const &,
                                            tensor_guid_t const &,
                                            std::vector<tensor_guid_t> const &);

/**
 * @brief Get the DLRM computation graph.
 *
 * @param DLRMConfig The config of DLRM model.
 * @return ComputationGraph The computation graph of a DLRM model.
 */
ComputationGraph get_dlrm_computation_graph(DLRMConfig const &);

} // namespace FlexFlow

#endif
