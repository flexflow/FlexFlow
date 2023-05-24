#include "op-attrs/ops/embedding.h"

namespace FlexFlow {

EmbeddingAttrs::EmbeddingAttrs(int _num_entries, int _out_channels, AggregateOp _aggr, DataType _data_type)
  : num_entries(_num_entries), out_channels(_out_channels), aggr(_aggr), data_type(_data_type)
{ }

}

