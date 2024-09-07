#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_SPLIT_TEST_SPLIT_TEST_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_SPLIT_TEST_SPLIT_TEST_H

#include "pcg/computation_graph.dtg.h"

namespace FlexFlow {

/**
 * @brief Get the computation graph of the old FlexFlow test model
 * <tt>split_test</tt>
 *
 * @note This is a tiny model developed for testing the original Unity
 * implementation. It is not a "real" model and has never been trained.
 */
ComputationGraph get_split_test_computation_graph(int batch_size);

} // namespace FlexFlow

#endif
