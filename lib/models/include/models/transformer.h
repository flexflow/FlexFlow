#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H

#include "models/transformer.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

// Helper functions to construct the Transformer model
tensor_guid_t create_transformer_feedforward_network(TransformerConfig const &,
                                                     ComputationGraphBuilder &,
                                                     tensor_guid_t const &);
tensor_guid_t create_transformer_encoder_layer(TransformerConfig const &,
                                               ComputationGraphBuilder &,
                                               tensor_guid_t const &);
tensor_guid_t create_transformer_decoder_layer(TransformerConfig const &,
                                               ComputationGraphBuilder &,
                                               tensor_guid_t const &,
                                               tensor_guid_t const &);

tensor_guid_t create_transformer_encoder(TransformerConfig const &,
                                         ComputationGraphBuilder &,
                                         tensor_guid_t const &);
tensor_guid_t create_transformer_decoder(TransformerConfig const &,
                                         ComputationGraphBuilder &,
                                         tensor_guid_t const &,
                                         tensor_guid_t const &);

/**
 * @brief Get the Transformer computation graph (PCG).
 *
 * @param TransformerConfig The config of Transformer model.
 * @return ComputationGraph The PCG of a Transformer model.
 */
ComputationGraph get_transformer_computation_graph(TransformerConfig const &);

} // namespace FlexFlow

#endif
