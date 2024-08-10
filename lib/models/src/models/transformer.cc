#include "models/transformer.h"

namespace FlexFlow {

ComputationGraph
    get_transformer_computation_graph(TransformerConfig const &tfConfig) {
  /**
   * @brief Helper to create an attention encoder layer
   */
  auto create_attention_encoder = [&](ComputationGraphBuilder *cgb,
                                      tensor_guid_t const &input,
                                      int hidden_dim,
                                      int num_heads,
                                      int kdim,
                                      int vdim) -> tensor_guid_t {
    tensor_guid_t t = cgb->multihead_attention(
        input, input, input, hidden_dim, num_heads, kdim, vdim);
    return cgb->dense(
        cgb->dense(t, hidden_dim, Activation::RELU, false /*bias*/),
        hidden_dim,
        Activation::RELU,
        false /*bias*/);
  };

  ComputationGraphBuilder cgb;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
          tfConfig.batch_size, tfConfig.sequence_length, tfConfig.hidden_size}},
      DataType::FLOAT,
  };

  tensor_guid_t input = cgb.create_tensor(input_shape, CreateGrad::YES);

  tensor_guid_t t = input;
  for (int i = 0; i < tfConfig.num_layers; i++) {
    t = create_attention_encoder(&cgb,
                                 t,
                                 tfConfig.hidden_size,
                                 tfConfig.num_heads,
                                 tfConfig.hidden_size / tfConfig.num_heads,
                                 tfConfig.hidden_size / tfConfig.num_heads);
  }
  t = cgb.dense(t, 1, Activation::RELU, false /*bias*/);

  return cgb.computation_graph;
}

} // namespace FlexFlow
