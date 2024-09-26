#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_BINARY_PARALLEL_SPLIT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_BINARY_PARALLEL_SPLIT_H

#include "compiler/series_parallel/pcg_binary_parallel_split.dtg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.dtg.h"

namespace FlexFlow {

PCGBinarySPDecomposition get_left_child(PCGBinaryParallelSplit const &);
PCGBinarySPDecomposition get_right_child(PCGBinaryParallelSplit const &);

} // namespace FlexFlow

#endif
