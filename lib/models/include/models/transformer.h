#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H

#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

struct TransformerConfig {
    int hidden_size = 1024;
    int embedding_size = 1024;
    int num_heads = 16;
    int num_layers = 12;
    int sequence_length = 512;
};

ComputationGraph get_transformer_computation_graph(TransformerConfig const &);

}  // namespace FlexFlow

#endif
