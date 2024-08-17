#include "models/transformer.h"

namespace FlexFlow {

ComputationGraph
    get_transformer_computation_graph(TransformerConfig const &tfConfig) {

  ComputationGraphBuilder cgb;
  int kdim = tfConfig.hidden_size / tfConfig.num_heads;
  int vdim = tfConfig.hidden_size / tfConfig.num_heads;
  std::vector<int> layer_norm_axis{2}; // Normalize the last dim

  /**
   * @brief Helper to create an attention encoder layer
   */
  auto create_attention_encoder =
      [&](tensor_guid_t const &input) -> tensor_guid_t {
    tensor_guid_t t = cgb.multihead_attention(input,
                                              input,
                                              input,
                                              tfConfig.hidden_size,
                                              tfConfig.num_heads,
                                              kdim,
                                              vdim);
    tensor_guid_t normalized_t = cgb.layer_norm(
        cgb.add(t, input), layer_norm_axis, true, tfConfig.layer_norm_eps);
    return cgb.layer_norm(cgb.add(normalized_t,
                                  cgb.dense(normalized_t,
                                            tfConfig.hidden_size,
                                            Activation::RELU,
                                            true)),
                          layer_norm_axis,
                          true,
                          tfConfig.layer_norm_eps);
  };

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
          tfConfig.batch_size, tfConfig.sequence_length, tfConfig.hidden_size}},
      DataType::FLOAT,
  };

  tensor_guid_t input = cgb.create_tensor(input_shape, CreateGrad::YES);
  tensor_guid_t t = input;
  for (int i = 0; i < tfConfig.num_layers; i++) {
    t = create_attention_encoder(t);
  }
  t = cgb.softmax(cgb.dense(t, 1, Activation::RELU, true));

  return cgb.computation_graph;
}

} // namespace FlexFlow
