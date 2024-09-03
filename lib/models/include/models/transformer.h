#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H

#include "models/transformer_config.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

// Helper functions to construct the Transformer model
tensor_guid_t create_transformer_feedforward_network(ComputationGraphBuilder &,
                                                     TransformerConfig const &,
                                                     tensor_guid_t const &);
tensor_guid_t create_transformer_encoder_layer(ComputationGraphBuilder &,
                                               TransformerConfig const &,
                                               tensor_guid_t const &);
tensor_guid_t create_transformer_decoder_layer(ComputationGraphBuilder &,
                                               TransformerConfig const &,
                                               tensor_guid_t const &,
                                               tensor_guid_t const &);

tensor_guid_t create_transformer_encoder(ComputationGraphBuilder &,
                                         TransformerConfig const &,
                                         tensor_guid_t const &);
tensor_guid_t create_transformer_decoder(ComputationGraphBuilder &,
                                         TransformerConfig const &,
                                         tensor_guid_t const &,
                                         tensor_guid_t const &);

TransformerConfig get_default_transformer_config();

/**
 * @brief Get the Transformer computation graph.
 *
 * @param TransformerConfig The config of Transformer model.
 * @return ComputationGraph The PCG of a Transformer model.
 */
ComputationGraph get_transformer_computation_graph(TransformerConfig const &);

} // namespace FlexFlow

#endif
