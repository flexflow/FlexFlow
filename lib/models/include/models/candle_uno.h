#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_CANDLE_UNO_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_CANDLE_UNO_H

// #include "candle_uno_config.dtg.h"
#include "pcg/computation_graph_builder.h"
#include <map>
#include <string>
#include <vector>

namespace FlexFlow {

struct CandleUnoConfig {
  size_t batch_size{};
  std::vector<int> dense_layers{};
  std::vector<int> dense_feature_layers{};
  std::map<std::string, int> feature_shapes{};
  std::map<std::string, std::string> input_features{};
};

// Helper functions to construct the Candle Uno model
tensor_guid_t create_candle_uno_feature_model(ComputationGraphBuilder &,
                                              CandleUnoConfig const &,
                                              tensor_guid_t const &);

/**
 * @brief Get the default configs of Candle Uno model.
 */
CandleUnoConfig get_default_candle_uno_config();

/**
 * @brief Get the Candle Uno computation graph.
 *
 * @param CandleUnoConfig The config of the Candle Uno model.
 * @return ComputationGraph The PCG of a Transformer model.
 */
ComputationGraph get_candle_uno_computation_graph(CandleUnoConfig const &);

} // namespace FlexFlow

#endif
