#include "models/transformer.h"

namespace FlexFlow {

tensor_guid_t create_feedforward_network(ComputationGraphBuilder &cgb,
                                         TransformerConfig const &config,
                                         tensor_guid_t const &input) {
  tensor_guid_t layer1_out = cgb.dense(
      input, config.dim_feedforward, Activation::RELU, /*use_bias=*/true);
  tensor_guid_t dropout_out = cgb.dropout(layer1_out, config.dropout);
  tensor_guid_t layer2_out = cgb.dense(dropout_out,
                                       config.num_features,
                                       /*activation=*/std::nullopt,
                                       /*use_bias=*/true);
  return cgb.dropout(layer2_out, config.dropout);
};

tensor_guid_t create_transformer_encoder_layer(ComputationGraphBuilder &cgb,
                                               TransformerConfig const &config,
                                               tensor_guid_t const &input) {
  std::vector<int> layer_norm_axis{2}; // Normalize the last dim
  int kdim = config.dim_feedforward / config.num_heads;
  int vdim = config.dim_feedforward / config.num_heads;
  tensor_guid_t self_attention = cgb.multihead_attention(input,
                                                         input,
                                                         input,
                                                         config.num_features,
                                                         config.num_heads,
                                                         kdim,
                                                         vdim,
                                                         config.dropout);
  tensor_guid_t normalized_t = cgb.layer_norm(cgb.add(self_attention, input),
                                              layer_norm_axis,
                                              /*elementwise_affine=*/true,
                                              config.layer_norm_eps);
  tensor_guid_t feedfoward_output =
      create_feedforward_network(cgb, config, normalized_t);
  return cgb.layer_norm(cgb.add(normalized_t, feedfoward_output),
                        layer_norm_axis,
                        /*elementwise_affine=*/true,
                        config.layer_norm_eps);
}

tensor_guid_t create_transformer_encoder(ComputationGraphBuilder &cgb,
                                         TransformerConfig const &config,
                                         tensor_guid_t const &input) {
  tensor_guid_t t = input;
  for (int i = 0; i < config.num_encoder_layers; i++) {
    t = create_transformer_encoder_layer(cgb, config, t);
  }
  return t;
};

tensor_guid_t
    create_transformer_decoder_layer(ComputationGraphBuilder &cgb,
                                     TransformerConfig const &config,
                                     tensor_guid_t const &input,
                                     tensor_guid_t const &encoder_output) {
  std::vector<int> layer_norm_axis{2}; // Normalize the last dim
  int kdim = config.dim_feedforward / config.num_heads;
  int vdim = config.dim_feedforward / config.num_heads;
  tensor_guid_t self_attention = cgb.multihead_attention(input,
                                                         input,
                                                         input,
                                                         config.num_features,
                                                         config.num_heads,
                                                         kdim,
                                                         vdim,
                                                         config.dropout);
  tensor_guid_t self_attention_normalized =
      cgb.layer_norm(cgb.add(input, self_attention),
                     layer_norm_axis,
                     /*elementwise_affine=*/true,
                     config.layer_norm_eps);

  tensor_guid_t mha = cgb.multihead_attention(input,
                                              encoder_output,
                                              encoder_output,
                                              config.num_features,
                                              config.num_heads,
                                              kdim,
                                              vdim,
                                              config.dropout);
  tensor_guid_t mha_normalized =
      cgb.layer_norm(cgb.add(self_attention_normalized, mha),
                     layer_norm_axis,
                     /*elementwise_affine=*/true,
                     config.layer_norm_eps);

  tensor_guid_t feedfoward_output =
      create_feedforward_network(cgb, config, mha_normalized);
  return cgb.layer_norm(cgb.add(mha_normalized, feedfoward_output),
                        layer_norm_axis,
                        /*elementwise_affine=*/true,
                        config.layer_norm_eps);
}

tensor_guid_t create_transformer_decoder(ComputationGraphBuilder &cgb,
                                         TransformerConfig const &config,
                                         tensor_guid_t const &input,
                                         tensor_guid_t const &encoder_output) {
  tensor_guid_t t = input;
  for (int i = 0; i < config.num_decoder_layers; i++) {
    t = create_transformer_decoder_layer(cgb, config, t, encoder_output);
  }
  return t;
}

ComputationGraph
    get_transformer_computation_graph(TransformerConfig const &config) {
  ComputationGraphBuilder cgb;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
          config.batch_size, config.sequence_length, config.num_features}},
      DataType::FLOAT,
  };
  tensor_guid_t input = cgb.create_tensor(input_shape, CreateGrad::YES);

  tensor_guid_t encoder_output = create_transformer_encoder(cgb, config, input);
  tensor_guid_t decoder_output =
      create_transformer_decoder(cgb, config, input, encoder_output);

  tensor_guid_t out_prob = cgb.softmax(cgb.dense(decoder_output,
                                                 /*outDim=*/config.vocab_size,
                                                 Activation::RELU,
                                                 /*use_bias=*/true));
  return cgb.computation_graph;
}

} // namespace FlexFlow
