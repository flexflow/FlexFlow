#include "models/bert/bert.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

BertConfig get_default_bert_config() {
  return BertConfig(/*vocab_size=*/30522,
                    /*hidden_size=*/768,
                    /*num_encoder_layers=*/12,
                    /*num_heads=*/12,
                    /*dim_feedforward=*/3072,
                    /*dropout=*/0.1,
                    /*layer_norm_eps=*/1e-12,
                    /*sequence_length=*/512,
                    /*batch_size=*/64);
}

tensor_guid_t create_feedforward_network(ComputationGraphBuilder &cgb,
                                         BertConfig const &config,
                                         tensor_guid_t const &input) {
  tensor_guid_t layer1_out = cgb.dense(
      input, config.dim_feedforward, Activation::GELU, /*use_bias=*/true);
  tensor_guid_t dropout_out = cgb.dropout(layer1_out, config.dropout);
  tensor_guid_t layer2_out = cgb.dense(dropout_out,
                                       config.hidden_size,
                                       /*activation=*/std::nullopt,
                                       /*use_bias=*/true);
  return cgb.dropout(layer2_out, config.dropout);
};

tensor_guid_t create_bert_encoder_layer(ComputationGraphBuilder &cgb,
                                        BertConfig const &config,
                                        tensor_guid_t const &input) {
  std::vector<int> layer_norm_axis = {2}; // Normalize the last dim
  int kdim = config.dim_feedforward / config.num_heads;
  int vdim = config.dim_feedforward / config.num_heads;
  tensor_guid_t self_attention = cgb.multihead_attention(input,
                                                         input,
                                                         input,
                                                         config.hidden_size,
                                                         config.num_heads,
                                                         kdim,
                                                         vdim,
                                                         config.dropout);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, self_attention));

  tensor_guid_t normalized = cgb.layer_norm(cgb.add(self_attention, input),
                                            layer_norm_axis,
                                            /*elementwise_affine=*/true,
                                            config.layer_norm_eps);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, normalized));

  tensor_guid_t feedforward_output =
      create_feedforward_network(cgb, config, normalized);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, feedforward_output));
  return cgb.layer_norm(cgb.add(normalized, feedforward_output),
                        layer_norm_axis,
                        /*elementwise_affine=*/true,
                        config.layer_norm_eps);
}

tensor_guid_t create_bert_encoder(ComputationGraphBuilder &cgb,
                                  BertConfig const &config,
                                  tensor_guid_t const &input) {
  tensor_guid_t t = input;
  for (int i = 0; i < config.num_encoder_layers; i++) {
    t = create_bert_encoder_layer(cgb, config, t);
  }
  return t;
};

ComputationGraph get_bert_computation_graph(BertConfig const &config) {
  ComputationGraphBuilder cgb;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
          config.batch_size, config.sequence_length, config.hidden_size}},
      DataType::FLOAT,
  };
  tensor_guid_t input = cgb.create_tensor(input_shape, CreateGrad::YES);

  tensor_guid_t encoder_output = create_bert_encoder(cgb, config, input);

  tensor_guid_t out_prob = cgb.softmax(cgb.dense(encoder_output,
                                                 /*outDim=*/config.vocab_size,
                                                 Activation::GELU,
                                                 /*use_bias=*/true));
  return cgb.computation_graph;
}

} // namespace FlexFlow
