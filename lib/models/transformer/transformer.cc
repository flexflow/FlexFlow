#include "transformer.h"

std::pair<FFConfig, TransformerConfig> getConfig() {
    FFConfig ffConfig;
    TransformerConfig tfConfig{};

    InputArgs const &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, tfConfig);

    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)", ffConfig.batchSize,
                  ffConfig.workersPerNode, ffConfig.numNodes);
    log_app.print("Hidden Size(%d)", tfConfig.hidden_size);
    log_app.print("Embedding Vocab Size(%d)", tfConfig.embedding_size);
    log_app.print("Number of Heads(%d)", tfConfig.num_heads);
    log_app.print("Number of Layers(%d)", tfConfig.num_layers);
    log_app.print("Sequence Length(%d)", tfConfig.sequence_length);

    return {ffconfig, tfConfig};
}

tensor_guid_t create_attention_encoder(ComputationGraphBuilder *cgb,
                                       tensor_guid_t const &input, int hidden_dim,
                                       int num_heads, int kdim, int vdim) {
    tensor_guid_t t =
        cgb->multihead_attention(input, input, input, hidden_dim, num_heads, kdim, vdim);
    return cgb->dense(cgb->dense(t, hidden_dim, Activation::RELU, false /*bias*/),
                      hidden_dim, Activation::None, false /*bias*/);
}

ComputationGraph constructComputationGraph() {
    auto [ffconfig, tfConfig] = getConfig();

    ComputationGraphBuilder cgb;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered<size_t>{ffConfig.batchSize, tfConfig.sequence_length,
                                     tfConfig.hidden_size}},
        DataType::FLOAT,
    };

    tensor_guid_t input = cgb.create_tensor(input_shape, CreateGrad::YES);

    auto t = input;
    for (int i = 0; i < tfConfig.num_layers; i++) {
        t = create_attention_encoder(&cgb, t, tfConfig.hidden_size, tfConfig.num_heads,
                                     tfConfig.hidden_size / tfConfig.num_heads,
                                     tfConfig.hidden_size / tfConfig.num_heads);
    }
    t = cgb.dense(t, 1, Activation::None, false /*bias*/);

    return cgb.computation_graph;
}

void parse_input_args(char **argv, int argc, TransformerConfig &config) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--num-layers")) {
            config.num_layers = atoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--embedding-size")) {
            config.embedding_size = atoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--hidden-size")) {
            config.hidden_size = atoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--num-heads")) {
            config.num_heads = atoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--sequence-length")) {
            config.sequence_length = atoi(argv[++i]);
            continue;
        }
    }
}
