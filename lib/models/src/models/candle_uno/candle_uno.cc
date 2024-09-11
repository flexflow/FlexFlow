#include "models/candle_uno/candle_uno.h"

namespace FlexFlow {

CandleUnoConfig get_default_candle_uno_config() {
  CandleUnoConfig config(
      /*batch_size=*/64,
      /*dense_layers=*/std::vector<int>(4, 4192),
      /*dense_feature_layers=*/std::vector<int>(8, 4192),
      /*feature_shapes=*/std::map<std::string, int>{},
      /*input_features=*/std::map<std::string, std::string>{},
      /*dropout=*/0.1);

  config.feature_shapes["dose"] = 1;
  config.feature_shapes["cell.rnaseq"] = 942;
  config.feature_shapes["drug.descriptors"] = 5270;
  config.feature_shapes["drug.fingerprints"] = 2048;

  config.input_features["dose1"] = "dose";
  config.input_features["dose2"] = "dose";
  config.input_features["cell.rnaseq"] = "cell.rnaseq";
  config.input_features["drug1.descriptors"] = "drug.descriptors";
  config.input_features["drug1.fingerprints"] = "drug.fingerprints";
  config.input_features["drug2.descriptors"] = "drug.descriptors";
  config.input_features["drug2.fingerprints"] = "drug.fingerprints";

  return config;
}

tensor_guid_t create_candle_uno_feature_model(ComputationGraphBuilder &cgb,
                                              CandleUnoConfig const &config,
                                              tensor_guid_t const &input) {
  tensor_guid_t t = input;
  for (auto const dense_dim : config.dense_feature_layers) {
    t = cgb.dense(t, dense_dim, Activation::RELU, /*use_bias=*/false);
  }
  return t;
}

ComputationGraph
    get_candle_uno_computation_graph(CandleUnoConfig const &config) {
  ComputationGraphBuilder cgb;

  auto create_input_tensor =
      [&](FFOrdered<size_t> const &dims) -> tensor_guid_t {
    TensorShape input_shape = TensorShape{
        TensorDims{dims},
        DataType::FLOAT,
    };
    return cgb.create_tensor(input_shape, CreateGrad::YES);
  };

  std::set<std::string> input_models;
  for (auto const &shape : config.feature_shapes) {
    auto const &type = shape.first;
    if (type.find(".") != std::string::npos) {
      std::string base_type = type.substr(0, type.find("."));
      if (base_type == "cell" || base_type == "drug") {
        input_models.insert(type);
      }
    }
  }

  std::vector<tensor_guid_t> all_inputs;
  std::vector<tensor_guid_t> encoded_inputs;

  for (auto const &input_feature : config.input_features) {
    std::string const &feature_name = input_feature.second;
    size_t shape = config.feature_shapes.at(feature_name);
    tensor_guid_t input = create_input_tensor({config.batch_size, shape});
    all_inputs.push_back(input);

    if (contains(input_models, feature_name)) {
      encoded_inputs.emplace_back(
          create_candle_uno_feature_model(cgb, config, input));
    } else {
      encoded_inputs.emplace_back(input);
    }
  }

  tensor_guid_t output = cgb.concat(encoded_inputs, /*axis=*/1);
  for (auto const &dense_layer_dim : config.dense_layers) {
    output = cgb.dense(
        output, dense_layer_dim, Activation::RELU, /*use_bias=*/false);
  }
  output =
      cgb.dense(output, 1, /*activation=*/std::nullopt, /*use_bias=*/false);

  return cgb.computation_graph;
}

} // namespace FlexFlow
