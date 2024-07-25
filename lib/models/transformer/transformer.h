#include "pcg/computation_graph_builder.h"

#define MAX_NUM_SAMPLES 65536

struct TransformerConfig {
    int hidden_size = 1024;
    int embedding_size = 1024;
    int num_heads = 16;
    int num_layers = 12;
    int sequence_length = 512;
};

ComputationGraph constructComputationGraph();
