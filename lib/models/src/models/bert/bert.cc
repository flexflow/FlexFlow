#include "models/bert/bert.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/computation_graph.h"
#include "pcg/initializers/truncated_normal_initializer_attrs.dtg.h"

namespace FlexFlow {

BertConfig get_default_bert_config() {
  return BertConfig{/*vocab_size=*/30522,
                    /*hidden_size=*/768,
                    /*num_encoder_layers=*/12,
                    /*num_heads=*/12,
                    /*dim_feedforward=*/3072,
                    /*hidden_act=*/Activation::GELU,
                    /*hidden_dropout_prob=*/0.1,
                    /*attention_probs_dropout_prob=*/0.1,
                    /*initializer_range=*/0.02,
                    /*layer_norm_eps=*/1e-12,
                    /*position_embedding_type=*/"absolute",
                    /*classifier_dropout=*/0.1,
                    /*sequence_length=*/512,
                    /*batch_size=*/64};
}

tensor_guid_t
    create_feedforward_network(ComputationGraphBuilder &cgb,
                               BertConfig const &config,
                               tensor_guid_t const &input,
                               InitializerAttrs const &bias_initializer,
                               InitializerAttrs const &projection_initializer) {
  tensor_guid_t layer1_out =
      cgb.dense(input,
                config.dim_feedforward,
                /*activation=*/config.hidden_act,
                /*use_bias=*/true,
                /*data_type=*/DataType::FLOAT,
                /*projection_initializer=*/projection_initializer,
                /*bias_initializer=*/bias_initializer);
  tensor_guid_t dropout_out =
      cgb.dropout(layer1_out, config.hidden_dropout_prob);
  tensor_guid_t layer2_out =
      cgb.dense(dropout_out,
                config.hidden_size,
                /*activation=*/std::nullopt,
                /*use_bias=*/true,
                /*data_type=*/DataType::FLOAT,
                /*projection_initializer=*/projection_initializer,
                /*bias_initializer=*/bias_initializer);
  return cgb.dropout(layer2_out, config.hidden_dropout_prob);
};

tensor_guid_t
    create_bert_encoder_layer(ComputationGraphBuilder &cgb,
                              BertConfig const &config,
                              tensor_guid_t const &input,
                              InitializerAttrs const &bias_initializer,
                              InitializerAttrs const &projection_initializer) {
  assert(num_dims(cgb.get_shape(input)) == 3);
  std::vector<int> layer_norm_axis = {2}; // Apply layernorm across the last dim
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
                              /*initializer=*/projection_initializer);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, self_attention));

  tensor_guid_t normalized = cgb.layer_norm(cgb.add(self_attention, input),
                                            layer_norm_axis,
                                            /*elementwise_affine=*/true,
                                            config.layer_norm_eps);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, normalized));

  tensor_guid_t feedforward_output = create_feedforward_network(
      cgb, config, normalized, bias_initializer, projection_initializer);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, feedforward_output));
  return cgb.layer_norm(cgb.add(normalized, feedforward_output),
                        layer_norm_axis,
                        /*elementwise_affine=*/true,
                        config.layer_norm_eps);
}

tensor_guid_t
    create_bert_encoder(ComputationGraphBuilder &cgb,
                        BertConfig const &config,
                        tensor_guid_t const &input,
                        InitializerAttrs const &bias_initializer,
                        InitializerAttrs const &projection_initializer) {
  tensor_guid_t t = input;
  for (int i = 0; i < config.num_encoder_layers; i++) {
    t = create_bert_encoder_layer(
        cgb, config, t, bias_initializer, projection_initializer);
  }
  return t;
};

ComputationGraph get_bert_computation_graph(BertConfig const &config) {
  if (config.position_embedding_type != "absolute") {
    throw mk_runtime_error(
        fmt::format("Currently only position_embedding_type=absolute is "
                    "supported, but found position_embedding_type={}. "
                    "If you need support this additional "
                    "position_embedding_type values, please create an issue.",
                    config.position_embedding_type));
  }

  ComputationGraphBuilder cgb;
  InitializerAttrs projection_initializer =
      InitializerAttrs{TruncatedNormalInitializerAttrs{
          /*seed=*/0,
          /*mean=*/0,
          /*stddev=*/config.initializer_range,
          /*min_cutoff=*/-2 * config.initializer_range,
          /*max_cutoff=*/2 * config.initializer_range}};
  InitializerAttrs bias_initializer = InitializerAttrs{ZeroInitializerAttrs{}};

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
          config.batch_size, config.sequence_length, config.hidden_size}},
      DataType::FLOAT,
  };
  tensor_guid_t input = cgb.create_input(input_shape, CreateGrad::YES);

  tensor_guid_t encoder_output = create_bert_encoder(
      cgb, config, input, bias_initializer, projection_initializer);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, encoder_output));

  tensor_guid_t out_prob =
      cgb.softmax(cgb.dense(encoder_output,
                            /*outDim=*/config.vocab_size,
                            /*activation=*/config.hidden_act,
                            /*use_bias=*/true,
                            /*data_type=*/DataType::FLOAT,
                            /*projection_initializer=*/projection_initializer,
                            /*bias_initializer=*/bias_initializer));
  assert(
      (cgb.get_shape(out_prob) ==
       TensorShape{
           TensorDims{FFOrdered<size_t>{
               config.batch_size, config.sequence_length, config.vocab_size}},
           DataType::FLOAT,
       }));

  return cgb.computation_graph;
}

} // namespace FlexFlow
