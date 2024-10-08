#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_PCG_BINARY_SERIES_SPLIT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_PCG_BINARY_SERIES_SPLIT_H

#include "compiler/series_parallel/pcg/pcg_binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_series_split.dtg.h"

namespace FlexFlow {

BinarySeriesSplit binary_series_split_from_pcg_series_split(PCGBinarySeriesSplit const &);

} // namespace FlexFlow

#endif
