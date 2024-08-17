#include "models/transformer.h"

namespace FlexFlow {

ComputationGraph
    get_transformer_computation_graph(TransformerConfig const &tfConfig) {
  ComputationGraphBuilder cgb;
  int kdim = tfConfig.hidden_size / tfConfig.num_heads;
  int vdim = tfConfig.hidden_size / tfConfig.num_heads;

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
    return cgb.dense(
        cgb.dense(t, tfConfig.hidden_size, Activation::RELU, false /*bias*/),
        tfConfig.hidden_size,
        std::nullopt,
        false /*bias*/);
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
  t = cgb.dense(t, 1, std::nullopt, false /*bias*/);

  return cgb.computation_graph;
}

} // namespace FlexFlow
