#ifndef _FLEXFLOW_EMBEDDING_ATTRS_H
#define _FLEXFLOW_EMBEDDING_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct EmbeddingAttrs : public UnaryOpAttrs {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  int num_entries, out_channels;
  AggrMode aggr;
  DataType data_type;
};

bool operator==(EmbeddingAttrs const &, EmbeddingAttrs const &);
bool operator<(EmbeddingAttrs const &, EmbeddingAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::EmbeddingAttrs, num_entries, out_channels, aggr, data_type);

namespace std {
template <>
struct hash<::FlexFlow::EmbeddingAttrs> {
  size_t operator()(::FlexFlow::EmbeddingAttrs const &) const;
};
} 

#endif 
