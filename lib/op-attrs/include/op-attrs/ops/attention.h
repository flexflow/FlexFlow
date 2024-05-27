#ifndef _FLEXFLOW_ATTENTION_ATTRS_H
#define _FLEXFLOW_ATTENTION_ATTRS_H

#include "core.h"
#include "op-attrs/ops/attention_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

int get_qProjSize(MultiHeadAttentionAttrs const &);
int get_vProjSize(MultiHeadAttentionAttrs const &);
int get_kProjSize(MultiHeadAttentionAttrs const &);
int get_oProjSize(MultiHeadAttentionAttrs const &);

int get_qSize(ParallelMultiHeadAttentionInputs const &);
int get_qSize(MultiHeadAttentionInputs const &);

int get_kSize(ParallelMultiHeadAttentionInputs const &);
int get_kSize(MultiHeadAttentionInputs const &);

int get_vSize(ParallelMultiHeadAttentionInputs const &);
int get_vSize(MultiHeadAttentionInputs const &);

int get_oSize(ParallelTensorShape const &);
int get_oSize(TensorShape const &);

int get_qoSeqLength(ParallelMultiHeadAttentionInputs const &);
int get_qoSeqLength(MultiHeadAttentionInputs const &);

int get_kvSeqLength(ParallelMultiHeadAttentionInputs const &);
int get_kvSeqLength(MultiHeadAttentionInputs const &);

int get_num_samples(ParallelMultiHeadAttentionInputs const &);
int get_num_samples(MultiHeadAttentionInputs const &);

tl::expected<TensorShape, std::string> get_weights_shape(MultiHeadAttentionAttrs const &,
                              TensorShape const &input_q,
                              TensorShape const &input_k,
                              TensorShape const &input_v);
tl::expected<ParallelTensorShape, std::string> get_weights_shape(MultiHeadAttentionAttrs const &,
                                      ParallelTensorShape const &input_q,
                                      ParallelTensorShape const &input_k,
                                      ParallelTensorShape const &input_v);

tl::expected<TensorShape, std::string> get_output_shape(MultiHeadAttentionAttrs const &,
                             TensorShape const &input_q,
                             TensorShape const &input_k,
                             TensorShape const &input_v);
tl::expected<ParallelTensorShape, std::string> get_output_shape(MultiHeadAttentionAttrs const &,
                                     ParallelTensorShape const &input_q,
                                     ParallelTensorShape const &input_k,
                                     ParallelTensorShape const &input_v);

CHECK_VALID_OP_ATTR(MultiHeadAttentionAttrs);
} // namespace FlexFlow

#endif
