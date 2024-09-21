#include "models/bert/bert.h"
#include "pcg/computation_graph.h"
#include "pcg/initializers/truncated_norm_initializer_attrs.dtg.h"

namespace FlexFlow {

BertConfig get_default_bert_config() {
  return BertConfig(/*vocab_size=*/30522,
                    /*hidden_size=*/768,
                    /*num_encoder_layers=*/12,
                    /*num_heads=*/12,
                    /*dim_feedforward=*/3072,
                    /*hidden_act=*/"GELU",
                    /*hidden_dropout_prob=*/0.1,
                    /*attention_probs_dropout_prob=*/0.1,
                    /*initializer_range=*/0.02,
                    /*layer_norm_eps=*/1e-12,
                    /*classifier_dropout=*/0.1,
                    /*sequence_length=*/512,
                    /*batch_size=*/64);
}

tensor_guid_t
    create_feedforward_network(ComputationGraphBuilder &cgb,
                               BertConfig const &config,
                               tensor_guid_t const &input,
                               Activation const &activation,
                               InitializerAttrs const &kernel_initializer) {
  tensor_guid_t layer1_out =
      cgb.dense(input,
                config.dim_feedforward,
                activation,
                /*use_bias=*/true,
                /*data_type=*/DataType::FLOAT,
                /*kernel_initializer=*/kernel_initializer);
  tensor_guid_t dropout_out =
      cgb.dropout(layer1_out, config.hidden_dropout_prob);
  tensor_guid_t layer2_out =
      cgb.dense(dropout_out,
                config.hidden_size,
                /*activation=*/std::nullopt,
                /*use_bias=*/true,
                /*data_type=*/DataType::FLOAT,
                /*kernel_initializer=*/kernel_initializer);
  return cgb.dropout(layer2_out, config.hidden_dropout_prob);
};

tensor_guid_t
    create_bert_encoder_layer(ComputationGraphBuilder &cgb,
                              BertConfig const &config,
                              tensor_guid_t const &input,
                              Activation const &activation,
                              InitializerAttrs const &kernel_initializer) {
  std::vector<int> layer_norm_axis = {2}; // Normalize the last dim
  int kdim = config.dim_feedforward / config.num_heads;
  int vdim = config.dim_feedforward / config.num_heads;
  tensor_guid_t self_attention =
      cgb.multihead_attention(input,
                              input,
                              input,
                              config.hidden_size,
                              config.num_heads,
                              kdim,
                              vdim,
                              /*dropout=*/config.attention_probs_dropout_prob,
                              /*bias=*/true,
                              /*add_bias_kv=*/false,
                              /*add_zero_attn=*/false,
                              /*initializer=*/kernel_initializer);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, self_attention));

  tensor_guid_t normalized = cgb.layer_norm(cgb.add(self_attention, input),
                                            layer_norm_axis,
                                            /*elementwise_affine=*/true,
                                            config.layer_norm_eps);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, normalized));

  tensor_guid_t feedforward_output = create_feedforward_network(
      cgb, config, normalized, activation, kernel_initializer);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, feedforward_output));
  return cgb.layer_norm(cgb.add(normalized, feedforward_output),
                        layer_norm_axis,
                        /*elementwise_affine=*/true,
                        config.layer_norm_eps);
}

tensor_guid_t create_bert_encoder(ComputationGraphBuilder &cgb,
                                  BertConfig const &config,
                                  tensor_guid_t const &input,
                                  Activation const &activation,
                                  InitializerAttrs const &kernel_initializer) {
  tensor_guid_t t = input;
  for (int i = 0; i < config.num_encoder_layers; i++) {
    t = create_bert_encoder_layer(
        cgb, config, t, activation, kernel_initializer);
  }
  return t;
};

ComputationGraph get_bert_computation_graph(BertConfig const &config) {

  auto get_activation_type = [&]() -> Activation {
    if (config.hidden_act == "GELU") {
      return Activation::GELU;
    } else if (config.hidden_act == "RELU") {
      return Activation::RELU;
    } else if (config.hidden_act == "SIGMOID") {
      return Activation::SIGMOID;
    } else if (config.hidden_act == "TANH") {
      return Activation::TANH;
    } else {
      throw mk_runtime_error("The given hidden_act is not supported. The "
                             "hidden_act string is: {}\n",
                             config.hidden_act);
    }
  };

  ComputationGraphBuilder cgb;
  Activation activation = get_activation_type();
  InitializerAttrs kernel_initializer = InitializerAttrs{
      TruncatedNormInitializerAttrs{/*seed=*/0,
                                    /*mean=*/0,
                                    /*stddev=*/config.initializer_range}};

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
          config.batch_size, config.sequence_length, config.hidden_size}},
      DataType::FLOAT,
  };
  tensor_guid_t input = cgb.create_tensor(input_shape, CreateGrad::YES);

  tensor_guid_t encoder_output =
      create_bert_encoder(cgb, config, input, activation, kernel_initializer);

  tensor_guid_t out_prob =
      cgb.softmax(cgb.dense(encoder_output,
                            /*outDim=*/config.vocab_size,
                            /*activation=*/activation,
                            /*use_bias=*/true,
                            /*data_type=*/DataType::FLOAT,
                            /*kernel_initializer=*/kernel_initializer));
  return cgb.computation_graph;
}

} // namespace FlexFlow
