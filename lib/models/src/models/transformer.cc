#include "models/transformer.h"

namespace FlexFlow {

tensor_guid_t
    Transformer::create_feedforward_network(tensor_guid_t const &input) {
  tensor_guid_t layer1_out = cgb_.dense(
      input, config_.dim_feedforward, Activation::RELU, true /* use_bias */);
  tensor_guid_t dropout_out = cgb_.dropout(layer1_out, config_.dropout);
  tensor_guid_t layer2_out = cgb_.dense(dropout_out,
                                        config_.num_features,
                                        std::nullopt /* activation */,
                                        true /* use_bias */);
  return cgb_.dropout(layer2_out, config_.dropout);
};

tensor_guid_t Transformer::create_encoder_layer(tensor_guid_t const &src) {
  std::vector<int> layer_norm_axis{2}; // Normalize the last dim
  tensor_guid_t self_attention = cgb_.multihead_attention(src,
                                                          src,
                                                          src,
                                                          config_.num_features,
                                                          config_.num_heads,
                                                          kdim_,
                                                          vdim_,
                                                          config_.dropout);
  tensor_guid_t normalized_t = cgb_.layer_norm(cgb_.add(self_attention, src),
                                               layer_norm_axis,
                                               true /* elementwise_affine */,
                                               config_.layer_norm_eps);
  tensor_guid_t feedforward_t = create_feedforward_network(normalized_t);
  return cgb_.layer_norm(cgb_.add(normalized_t, feedforward_t),
                         layer_norm_axis,
                         true /* elementwise_affine */,
                         config_.layer_norm_eps);
}

tensor_guid_t Transformer::create_encoder(tensor_guid_t const &src) {
  tensor_guid_t t = src;
  for (int i = 0; i < config_.num_encoder_layers; i++) {
    t = create_encoder_layer(t);
  }
  return t;
};

tensor_guid_t
    Transformer::create_decoder_layer(tensor_guid_t const &tgt,
                                      tensor_guid_t const &encoder_output) {
  std::vector<int> layer_norm_axis{2}; // Normalize the last dim
  tensor_guid_t self_attention = cgb_.multihead_attention(tgt,
                                                          tgt,
                                                          tgt,
                                                          config_.num_features,
                                                          config_.num_heads,
                                                          kdim_,
                                                          vdim_,
                                                          config_.dropout);
  tensor_guid_t self_attention_normalized =
      cgb_.layer_norm(cgb_.add(tgt, self_attention),
                      layer_norm_axis,
                      true /* elementwise_affine */,
                      config_.layer_norm_eps);

  tensor_guid_t mha = cgb_.multihead_attention(tgt,
                                               encoder_output,
                                               encoder_output,
                                               config_.num_features,
                                               config_.num_heads,
                                               kdim_,
                                               vdim_,
                                               config_.dropout);
  tensor_guid_t mha_normalized =
      cgb_.layer_norm(cgb_.add(self_attention_normalized, mha),
                      layer_norm_axis,
                      true /* elementwise_affine */,
                      config_.layer_norm_eps);

  tensor_guid_t feedforward_t = create_feedforward_network(mha_normalized);
  return cgb_.layer_norm(cgb_.add(mha_normalized, feedforward_t),
                         layer_norm_axis,
                         true /* elementwise_affine */,
                         config_.layer_norm_eps);
}

tensor_guid_t Transformer::create_decoder(tensor_guid_t const &tgt,
                                          tensor_guid_t const &encoder_output) {
  tensor_guid_t t = tgt;
  for (int i = 0; i < config_.num_decoder_layers; i++) {
    t = create_decoder_layer(t, encoder_output);
  }
  return t;
}

void Transformer::init_model() {
  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
          config_.batch_size, config_.sequence_length, config_.num_features}},
      DataType::FLOAT,
  };

  tensor_guid_t input = cgb_.create_tensor(input_shape, CreateGrad::YES);

  tensor_guid_t encoder_output = create_encoder(input);
  tensor_guid_t decoder_output = create_decoder(input, encoder_output);

  tensor_guid_t out_prob =
      cgb_.softmax(cgb_.dense(decoder_output,
                              config_.vocab_size /* outDim */,
                              Activation::RELU,
                              true /* bias */));
}

ComputationGraph Transformer::get_computation_graph() const {
  return cgb_.computation_graph;
}

ComputationGraph
    get_transformer_computation_graph(TransformerConfig const &tfConfig) {
  return Transformer(tfConfig).get_computation_graph();
}

} // namespace FlexFlow
