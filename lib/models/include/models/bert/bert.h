#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_BERT_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_BERT_H

#include "models/bert/bert_config.dtg.h"
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
 *
 * @details Refer to
 * https://huggingface.co/docs/transformers/v4.18.0/en/model_doc/bert#transformers.BertConfig
 * for default configs.
 */
BertConfig get_default_bert_config();

/**
 * @brief Get the BERT computation graph.
 *
 * @note This is a plain encoder-only model for pre-training.
 *
 * @param BertConfig The config of BERT model.
 * @return ComputationGraph The computation graph of a BERT model.
 */
ComputationGraph get_bert_computation_graph(BertConfig const &);

} // namespace FlexFlow

#endif
