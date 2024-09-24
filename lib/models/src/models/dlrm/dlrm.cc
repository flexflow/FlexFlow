#include "models/dlrm/dlrm.h"
#include "pcg/computation_graph.h"
#include "utils/containers/concat_vectors.h"

namespace FlexFlow {

DLRMConfig get_default_dlrm_config() {
  DLRMConfig config{/*sparse_feature_size=*/64,
                    /*sigmoid_bot=*/-1,
                    /*sigmoid_top=*/-1,
                    /*embedding_bag_size=*/1,
                    /*loss_threshold=*/0,
                    /*embedding_size=*/std::vector<int>{},
                    /*mlp_bot=*/std::vector<size_t>{},
                    /*mlp_top=*/std::vector<size_t>{},
                    /*arch_interaction_op=*/"cat",
                    /*dataset_path=*/"",
                    /*data_size=*/-1,
                    /*batch_size=*/64};

  config.embedding_size.emplace_back(1000000);
  config.embedding_size.emplace_back(1000000);
  config.embedding_size.emplace_back(1000000);
  config.embedding_size.emplace_back(1000000);

  config.mlp_bot.emplace_back(4);
  config.mlp_bot.emplace_back(64);
  config.mlp_bot.emplace_back(64);

  config.mlp_top.emplace_back(64);
  config.mlp_top.emplace_back(64);
  config.mlp_top.emplace_back(2);

  return config;
}

tensor_guid_t create_dlrm_mlp(ComputationGraphBuilder &cgb,
                              DLRMConfig const &config,
                              tensor_guid_t const &input,
                              std::vector<size_t> const &mlp_layers,
                              int const &sigmoid_layer) {
  tensor_guid_t t = input;
  for (size_t i = 0; i < mlp_layers.size() - 1; i++) {
    float std_dev = sqrt(2.0f / (mlp_layers[i + 1] + mlp_layers[i]));
    InitializerAttrs projection_initializer =
        InitializerAttrs{NormInitializerAttrs{
            /*seed=*/std::rand(),
            /*mean=*/0,
            /*stddev=*/std_dev,
        }};

    std_dev = sqrt(2.0f / mlp_layers[i + 1]);
    InitializerAttrs bias_initializer = InitializerAttrs{NormInitializerAttrs{
        /*seed=*/std::rand(),
        /*mean=*/0,
        /*stddev=*/std_dev,
    }};

    Activation activation =
        (i == sigmoid_layer) ? Activation::SIGMOID : Activation::RELU;

    t = cgb.dense(/*input=*/t,
                  /*outDim=*/mlp_layers[i + 1],
                  /*activation=*/activation,
                  /*use_bias=*/true,
                  /*data_type=*/DataType::FLOAT,
                  /*projection_initializer=*/projection_initializer,
                  /*bias_initializer=*/bias_initializer);
  }
  return t;
}

tensor_guid_t create_dlrm_emb(ComputationGraphBuilder &cgb,
                              DLRMConfig const &config,
                              tensor_guid_t const &input,
                              int const &input_dim,
                              int const &output_dim) {
  float range = sqrt(1.0f / input_dim);
  InitializerAttrs embed_initializer = InitializerAttrs{UniformInitializerAttrs{
      /*seed=*/std::rand(),
      /*min_val=*/-range,
      /*max_val=*/range,
  }};

  tensor_guid_t t = cgb.embedding(input,
                                  /*num_entries=*/input_dim,
                                  /*outDim=*/output_dim,
                                  /*aggr=*/AggregateOp::SUM,
                                  /*dtype=*/DataType::HALF,
                                  /*kernel_initializer=*/embed_initializer);
  return cgb.cast(t, DataType::FLOAT);
}

tensor_guid_t create_dlrm_interact_features(
    ComputationGraphBuilder &cgb,
    DLRMConfig const &config,
    tensor_guid_t const &bottom_mlp_output,
    std::vector<tensor_guid_t> const &emb_outputs) {
  if (config.arch_interaction_op != "cat") {
    throw mk_runtime_error(fmt::format(
        "Currently only arch_interaction_op=cat is supported, but found "
        "arch_interaction_op={}. If you need support for additional "
        "arch_interaction_op value, please create an issue.",
        config.arch_interaction_op));
  }

  return cgb.concat(
      /*tensors=*/concat_vectors({bottom_mlp_output}, emb_outputs),
      /*axis=*/1);
}

ComputationGraph get_dlrm_computation_graph(DLRMConfig const &config) {
  ComputationGraphBuilder cgb;

  auto create_input_tensor = [&](FFOrdered<size_t> const &dims,
                                 DataType const &data_type) -> tensor_guid_t {
    TensorShape input_shape = TensorShape{
        TensorDims{dims},
        data_type,
    };
    return cgb.create_input(input_shape, CreateGrad::YES);
  };

  // Create input tensors
  std::vector<tensor_guid_t> sparse_inputs(
      config.embedding_size.size(),
      create_input_tensor({config.batch_size, config.embedding_bag_size},
                          DataType::INT64));

  tensor_guid_t dense_input = create_input_tensor(
      {config.batch_size, config.mlp_bot.front()},
      DataType::HALF); // TODO: change this to DataType::FLOAT after cgb.cast is
                       // implemented.

  // Construct the model
  tensor_guid_t bottom_mlp_output = create_dlrm_mlp(
      cgb, config, dense_input, config.mlp_bot, config.sigmoid_bot);

  std::vector<tensor_guid_t> emb_outputs;
  for (size_t i = 0; i < config.embedding_size.size(); i++) {
    int input_dim = config.embedding_size[i];
    int output_dim = config.sparse_feature_size;
    emb_outputs.emplace_back(
        create_dlrm_emb(cgb, config, sparse_inputs[i], input_dim, output_dim));
  }

  tensor_guid_t interacted_features = create_dlrm_interact_features(
      cgb, config, bottom_mlp_output, emb_outputs);

  tensor_guid_t output = create_dlrm_mlp(cgb,
                                         config,
                                         interacted_features,
                                         config.mlp_top,
                                         config.mlp_top.size() - 2);

  return cgb.computation_graph;
}

} // namespace FlexFlow
