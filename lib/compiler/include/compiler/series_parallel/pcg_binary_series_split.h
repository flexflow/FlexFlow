#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_BINARY_SERIES_SPLIT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_BINARY_SERIES_SPLIT_H

#include "compiler/series_parallel/pcg_binary_series_split.dtg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_series_split.dtg.h"

namespace FlexFlow {

BinarySeriesSplit get_raw_graph_series_split(PCGBinarySeriesSplit const &);

PCGBinarySPDecomposition get_left_child(PCGBinarySeriesSplit const &);
PCGBinarySPDecomposition get_right_child(PCGBinarySeriesSplit const &);

} // namespace FlexFlow

#endif
