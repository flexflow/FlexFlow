#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_PCG_BINARY_PARALLEL_SPLIT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_PCG_BINARY_PARALLEL_SPLIT_H

#include "compiler/series_parallel/pcg/pcg_binary_parallel_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_parallel_split.dtg.h"

namespace FlexFlow {

BinaryParallelSplit binary_parallel_split_from_pcg_parallel_split(PCGBinaryParallelSplit const &);

} // namespace FlexFlow

#endif
