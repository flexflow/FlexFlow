#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_BERT_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_BERT_H

#include "models/bert_config.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

// Helper functions to construct the BERT model
tensor_guid_t create_bert_feedforward_network(ComputationGraphBuilder &,
                                              BertConfig const &,
                                              tensor_guid_t const &);
tensor_guid_t create_bert_encoder_layer(ComputationGraphBuilder &,
                                        BertConfig const &,
                                        tensor_guid_t const &);
tensor_guid_t create_bert_encoder(ComputationGraphBuilder &,
                                  BertConfig const &,
                                  tensor_guid_t const &);

/**
 * @brief Get the base config of the BERT model.
 */
BertConfig get_default_bert_config();

/**
 * @brief Get the BERT computation graph.
 *
 * @param BertConfig The config of BERT model.
 * @return ComputationGraph The PCG of a BERT model.
 */
ComputationGraph get_bert_computation_graph(BertConfig const &);

} // namespace FlexFlow

#endif
