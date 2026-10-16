#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ATTENTION_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ATTENTION_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/attention/multihead_attention_inputs.dtg.h"
#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.dtg.h"
#include "op-attrs/ops/attention_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"

namespace FlexFlow {

positive_int get_qProjSize(MultiHeadAttentionAttrs const &);
positive_int get_vProjSize(MultiHeadAttentionAttrs const &);
positive_int get_kProjSize(MultiHeadAttentionAttrs const &);
positive_int get_oProjSize(MultiHeadAttentionAttrs const &);

positive_int get_qSize(MultiHeadAttentionParallelInputs const &);
positive_int get_qSize(MultiHeadAttentionInputs const &);

positive_int get_kSize(MultiHeadAttentionParallelInputs const &);
positive_int get_kSize(MultiHeadAttentionInputs const &);

positive_int get_vSize(MultiHeadAttentionParallelInputs const &);
positive_int get_vSize(MultiHeadAttentionInputs const &);

positive_int get_oSize(ParallelTensorShape const &);
positive_int get_oSize(TensorShape const &);

positive_int get_qoSeqLength(MultiHeadAttentionParallelInputs const &);
positive_int get_qoSeqLength(MultiHeadAttentionInputs const &);

positive_int get_kvSeqLength(MultiHeadAttentionParallelInputs const &);
positive_int get_kvSeqLength(MultiHeadAttentionInputs const &);

positive_int get_num_samples(MultiHeadAttentionParallelInputs const &);
positive_int get_num_samples(MultiHeadAttentionInputs const &);

std::map<TensorSlotName, IncomingTensorRole>
    get_attention_incoming_tensor_roles(MultiHeadAttentionAttrs const &);

TensorShape get_weights_shape(MultiHeadAttentionAttrs const &,
                              TensorShape const &input_q,
                              TensorShape const &input_k,
                              TensorShape const &input_v);
TensorShape get_input_bias_shape(MultiHeadAttentionAttrs const &,
                                 TensorShape const &input_q,
                                 TensorShape const &input_k,
                                 TensorShape const &input_v);
TensorShape get_output_bias_shape(MultiHeadAttentionAttrs const &,
                                  TensorShape const &input_q,
                                  TensorShape const &input_k,
                                  TensorShape const &input_v);
TensorShape get_output_shape(MultiHeadAttentionAttrs const &,
                             TensorShape const &input_q,
                             TensorShape const &input_k,
                             TensorShape const &input_v);

std::map<TensorSlotName, TensorShape>
    get_weight_shapes(MultiHeadAttentionAttrs const &,
                      TensorShape const &input_q,
                      TensorShape const &input_k,
                      TensorShape const &input_v);

ParallelTensorDims
    get_weights_parallel_dims(MultiHeadAttentionAttrs const &,
                              ParallelTensorShape const &input_q,
                              ParallelTensorShape const &input_k,
                              ParallelTensorShape const &input_v);
ParallelTensorDims
    get_input_bias_parallel_dims(MultiHeadAttentionAttrs const &,
                                 ParallelTensorShape const &input_q,
                                 ParallelTensorShape const &input_k,
                                 ParallelTensorShape const &input_v);
ParallelTensorDims
    get_output_bias_parallel_dims(MultiHeadAttentionAttrs const &,
                                  ParallelTensorShape const &input_q,
                                  ParallelTensorShape const &input_k,
                                  ParallelTensorShape const &input_v);

ParallelTensorShape get_weights_shape(MultiHeadAttentionAttrs const &,
                                      ParallelTensorShape const &input_q,
                                      ParallelTensorShape const &input_k,
                                      ParallelTensorShape const &input_v);
ParallelTensorShape get_input_bias_shape(MultiHeadAttentionAttrs const &,
                                         ParallelTensorShape const &input_q,
                                         ParallelTensorShape const &input_k,
                                         ParallelTensorShape const &input_v);
ParallelTensorShape get_output_bias_shape(MultiHeadAttentionAttrs const &,
                                          ParallelTensorShape const &input_q,
                                          ParallelTensorShape const &input_k,
                                          ParallelTensorShape const &input_v);
ParallelTensorShape get_output_shape(MultiHeadAttentionAttrs const &,
                                     ParallelTensorShape const &input_q,
                                     ParallelTensorShape const &input_k,
                                     ParallelTensorShape const &input_v);

std::map<TensorSlotName, ParallelTensorShape>
    get_weight_shapes(MultiHeadAttentionAttrs const &,
                      ParallelTensorShape const &input_q,
                      ParallelTensorShape const &input_k,
                      ParallelTensorShape const &input_v);

std::map<TensorSlotName, InitializerAttrs> get_initializers(
    MultiHeadAttentionAttrs const &,
    TensorShape const &input_q,
    TensorShape const &input_k,
    TensorShape const &input_v,
    std::optional<InitializerAttrs> const &weights_initializer = std::nullopt,
    std::optional<InitializerAttrs> const &input_bias_initializer =
        std::nullopt,
    std::optional<InitializerAttrs> const &output_bias_initializer =
        std::nullopt);

} // namespace FlexFlow

#endif
