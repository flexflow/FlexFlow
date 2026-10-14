#include "models/bert/bert.h"
#include "op-attrs/initializers/truncated_normal_initializer_attrs.dtg.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/computation_graph.h"
#include "utils/exception.h"

namespace FlexFlow {

BertConfig get_default_bert_config() {
  return BertConfig{
      /*vocab_size=*/30522_p,
      /*hidden_size=*/768_p,
      /*num_encoder_layers=*/12_p,
      /*num_heads=*/12_p,
      /*dim_feedforward=*/3072_p,
      /*hidden_act=*/Activation::GELU,
      /*hidden_dropout_prob=*/0.1,
      /*attention_probs_dropout_prob=*/0.1,
      /*initializer_range=*/0.02,
      /*layer_norm_eps=*/1e-12,
      /*position_embedding_type=*/"absolute",
      /*classifier_dropout=*/0.1,
      /*sequence_length=*/512_p,
      /*batch_size=*/64_p,
  };
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
  ASSERT(get_num_dims(cgb.get_shape(input).dims) == 3);
  std::set<relative_ff_dim_t> layer_norm_axis = {
      relative_ff_dim_t{-1}}; // Apply layernorm across the last dim
  positive_int kdim = positive_int{config.dim_feedforward / config.num_heads};
  positive_int vdim = positive_int{config.dim_feedforward / config.num_heads};
  tensor_guid_t self_attention =
      cgb.multihead_attention(/*query=*/input,
                              /*key=*/input,
                              /*value=*/input,
                              /*embed_dim=*/config.hidden_size,
                              /*num_heads=*/config.num_heads,
                              /*kdim=*/kdim,
                              /*vdim=*/vdim,
                              /*dropout=*/config.attention_probs_dropout_prob,
                              /*bias=*/true,
                              /*add_bias_kv=*/false,
                              /*add_zero_attn=*/false,
                              /*initializer=*/projection_initializer);
  ASSERT(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, self_attention));

  tensor_guid_t normalized = cgb.layer_norm(cgb.add(self_attention, input),
                                            layer_norm_axis,
                                            /*elementwise_affine=*/true,
                                            config.layer_norm_eps);
  ASSERT(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, normalized));

  tensor_guid_t feedforward_output = create_feedforward_network(
      cgb, config, normalized, bias_initializer, projection_initializer);
  ASSERT(are_tensor_guid_shapes_equivalent(
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
    PANIC(fmt::format("Currently only position_embedding_type=absolute is "
                      "supported, but found position_embedding_type={}. "
                      "If you need support for additional "
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
      TensorDims{FFOrdered<positive_int>{
          config.batch_size, config.sequence_length, config.hidden_size}},
      DataType::FLOAT,
  };
  tensor_guid_t input = cgb.create_input(input_shape, CreateGrad::YES);

  tensor_guid_t encoder_output = create_bert_encoder(
      cgb, config, input, bias_initializer, projection_initializer);
  ASSERT(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, encoder_output));

  tensor_guid_t out_prob =
      cgb.softmax(cgb.dense(encoder_output,
                            /*outDim=*/config.vocab_size,
                            /*activation=*/config.hidden_act,
                            /*use_bias=*/true,
                            /*data_type=*/DataType::FLOAT,
                            /*projection_initializer=*/projection_initializer,
                            /*bias_initializer=*/bias_initializer));
  ASSERT(
      (cgb.get_shape(out_prob) ==
       TensorShape{
           TensorDims{FFOrdered<positive_int>{
               config.batch_size, config.sequence_length, config.vocab_size}},
           DataType::FLOAT,
       }));

  return cgb.computation_graph;
}

} // namespace FlexFlow
