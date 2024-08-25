#include "models/transformer.h"

namespace FlexFlow {

void Transformer::init_model() {
  int kdim = config_.hidden_size / config_.num_heads;
  int vdim = config_.hidden_size / config_.num_heads;
  std::vector<int> layer_norm_axis{2}; // Normalize the last dim

  /**
   * @brief Helper to create an attention encoder layer
   */
  auto create_attention_encoder =
      [&](tensor_guid_t const &query,
          tensor_guid_t const &key,
          tensor_guid_t const &value) -> tensor_guid_t {
    tensor_guid_t t = cgb_.multihead_attention(
        query, key, value, config_.hidden_size, config_.num_heads, kdim, vdim);
    tensor_guid_t normalized_t = cgb_.layer_norm(cgb_.add(t, query),
                                                 layer_norm_axis,
                                                 true /* elementwise_affine */,
                                                 config_.layer_norm_eps);
    return cgb_.layer_norm(cgb_.add(normalized_t,
                                    cgb_.dense(normalized_t,
                                               config_.hidden_size,
                                               Activation::RELU,
                                               true /* bias */)),
                           layer_norm_axis,
                           true /* elementwise_affine */,
                           config_.layer_norm_eps);
  };

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
          config_.batch_size, config_.sequence_length, config_.hidden_size}},
      DataType::FLOAT,
  };

  tensor_guid_t input = cgb_.create_tensor(input_shape, CreateGrad::YES);
  tensor_guid_t t = input;
  tensor_guid_t key = input;
  tensor_guid_t value = input;
  for (int i = 0; i < config_.num_layers; i++) {
    t = create_attention_encoder(t, key, value);
  }
  t = cgb_.softmax(cgb_.dense(
      t, config_.vocab_size /* outDim */, Activation::RELU, true /* bias */));
}

ComputationGraph Transformer::get_computation_graph() const {
  return cgb_.computation_graph;
}

ComputationGraph
    get_transformer_computation_graph(TransformerConfig const &tfConfig) {
  return Transformer(tfConfig).get_computation_graph();
}

} // namespace FlexFlow
