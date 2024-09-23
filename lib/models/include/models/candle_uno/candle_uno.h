#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_CANDLE_UNO_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_CANDLE_UNO_H

#include "candle_uno_config.dtg.h"
#include "pcg/computation_graph_builder.h"
#include <map>
#include <string>
#include <vector>

namespace FlexFlow {

// Helper functions to construct the Candle Uno model
tensor_guid_t create_candle_uno_feature_model(ComputationGraphBuilder &,
                                              CandleUnoConfig const &,
                                              tensor_guid_t const &);

/**
 * @brief Get the default configs of Candle Uno model.
 *
 * @details The default configs come from the dataset used by the original
 * model:
 * https://github.com/ECP-CANDLE/Benchmarks/tree/f6a3da8818308c9edcd9fedbcf831dd5968efcdd/Pilot1/Uno
 */
CandleUnoConfig get_default_candle_uno_config();

/**
 * @brief Get the Candle Uno computation graph.
 *
 * @details CandleUnoConfig.feature_shapes is a map from feature name to the
 * number of channels for the feature, and CandleUnoConfig.input_features is a
 * map from specific data identifier in the dataset to the feature name used in
 * this model.
 *
 * @param CandleUnoConfig The config of the Candle Uno model.
 * @return ComputationGraph The PCG of a Transformer model.
 */
ComputationGraph get_candle_uno_computation_graph(CandleUnoConfig const &);

} // namespace FlexFlow

#endif
