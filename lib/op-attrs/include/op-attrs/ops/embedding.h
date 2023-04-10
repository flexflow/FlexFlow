#ifndef _FLEXFLOW_EMBEDDING_ATTRS_H
#define _FLEXFLOW_EMBEDDING_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "utils/visitable.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

struct EmbeddingAttrs {
public:
  EmbeddingAttrs(int num_entries, int out_channels, AggrMode aggr, DataType data_type);
public:
  int num_entries, out_channels;
  AggrMode aggr;
  DataType data_type;
};

bool operator==(EmbeddingAttrs const &, EmbeddingAttrs const &);
bool operator<(EmbeddingAttrs const &, EmbeddingAttrs const &);

TensorShape get_weights_shape(EmbeddingAttrs const &, TensorShape const &);

}

VISITABLE_STRUCT(::FlexFlow::EmbeddingAttrs, num_entries, out_channels, aggr, data_type);

namespace std {
template <>
struct hash<::FlexFlow::EmbeddingAttrs> {
  size_t operator()(::FlexFlow::EmbeddingAttrs const &) const;
};
} 

#endif 
