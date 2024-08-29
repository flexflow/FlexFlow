#include "models/candle_uno.h"

namespace FlexFlow {

tensor_guid_t CandleUno::build_feature_model(tensor_guid_t const &input) {
  tensor_guid_t t = input;
  for (auto const dense_dim : config_.dense_feature_layers) {
    t = cgb_.dense(t, dense_dim, Activation::RELU, false /* use_bias */);
  }
  return t;
}

void CandleUno::init_model() {
  auto create_input_tensor = [&](FFOrdered<size_t> dims) -> tensor_guid_t {
    TensorShape input_shape = TensorShape{
        TensorDims{dims},
        DataType::FLOAT,
    };
    return cgb_.create_tensor(input_shape, CreateGrad::YES);
  };

  std::set<std::string> input_models;
  for (auto const &shape : config_.feature_shapes) {
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

  for (auto const &input_feature : config_.input_features) {
    auto const &feature_name = input_feature.second;
    assert(config_.feature_shapes.find(feature_name) !=
           config_.feature_shapes.end());

    size_t shape = config_.feature_shapes[feature_name];
    tensor_guid_t input = create_input_tensor({config_.batch_size, shape});
    all_inputs.push_back(input);

    if (input_models.find(feature_name) != input_models.end()) {
      encoded_inputs.emplace_back(build_feature_model(input));
    } else {
      encoded_inputs.emplace_back(input);
    }
  }

  tensor_guid_t output =
      cgb_.concat(encoded_inputs.size(), encoded_inputs, -1 /* axis */);
  for (auto const dense_layer_dim : config_.dense_layers) {
    output = cgb_.dense(
        output, dense_layer_dim, Activation::RELU, false /* use_bias */);
  }
  output = cgb_.dense(
      output, 1, std::nullopt /* activation */, false /* use_bias */);
}

ComputationGraph CandleUno::get_computation_graph() const {
  return cgb_.computation_graph;
}

ComputationGraph
    get_candle_uno_computation_graph(CandleUnoConfig const &config) {
  return CandleUno(config).get_computation_graph();
}

} // namespace FlexFlow
