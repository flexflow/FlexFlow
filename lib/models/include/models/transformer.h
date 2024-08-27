#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H

#include "models/transformer.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

class Transformer {
public:
  Transformer(TransformerConfig const &config)
      : config_(config), kdim_(config_.dim_feedforward / config_.num_heads),
        vdim_(config_.dim_feedforward / config_.num_heads) {
    init_model();
  }

  [[nodiscard]] ComputationGraph get_computation_graph() const;

private:
  void init_model();

  tensor_guid_t create_feedforward_network(tensor_guid_t const &);

  tensor_guid_t create_encoder_layer(tensor_guid_t const &);
  tensor_guid_t create_encoder(tensor_guid_t const &);

  tensor_guid_t create_decoder_layer(tensor_guid_t const &,
                                     tensor_guid_t const &);
  tensor_guid_t create_decoder(tensor_guid_t const &, tensor_guid_t const &);

private:
  TransformerConfig config_;
  ComputationGraphBuilder cgb_;
  int kdim_;
  int vdim_;
};

ComputationGraph get_transformer_computation_graph(TransformerConfig const &);

} // namespace FlexFlow

#endif
