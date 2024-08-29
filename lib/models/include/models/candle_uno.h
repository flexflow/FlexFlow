#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_CANDLE_UNO_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_CANDLE_UNO_H

#include "pcg/computation_graph_builder.h"
#include <map>
#include <string>
#include <vector>

namespace FlexFlow {

struct CandleUnoConfig {
  CandleUnoConfig() : dense_layers(4, 4192), dense_feature_layers(8, 4192) {
    feature_shapes["dose"] = 1;
    feature_shapes["cell.rnaseq"] = 942;
    feature_shapes["drug.descriptors"] = 5270;
    feature_shapes["drug.fingerprints"] = 2048;

    input_features["dose1"] = "dose";
    input_features["dose2"] = "dose";
    input_features["cell.rnaseq"] = "cell.rnaseq";
    input_features["drug1.descriptors"] = "drug.descriptors";
    input_features["drug1.fingerprints"] = "drug.fingerprints";
    input_features["drug2.descriptors"] = "drug.descriptors";
    input_features["drug2.fingerprints"] = "drug.fingerprints";
  }

  std::vector<int> dense_layers, dense_feature_layers;
  std::map<std::string, int> feature_shapes;
  std::map<std::string, std::string> input_features;
  size_t batch_size{};
};

class CandleUno {
public:
  static const size_t max_num_samples = 4196;

public:
  CandleUno(CandleUnoConfig const &config) : config_(config) {
    init_model();
  }

  [[nodiscard]] ComputationGraph get_computation_graph() const;

private:
  void init_model();

  tensor_guid_t build_feature_model(tensor_guid_t const &);

private:
  CandleUnoConfig config_;
  ComputationGraphBuilder cgb_;
};

ComputationGraph get_candle_uno_computation_graph(CandleUnoConfig const &);

} // namespace FlexFlow

#endif
