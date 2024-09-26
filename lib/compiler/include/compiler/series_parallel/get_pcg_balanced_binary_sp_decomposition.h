#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_GET_PCG_BALANCED_BINARY_SP_DECOMPOSITION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_GET_PCG_BALANCED_BINARY_SP_DECOMPOSITION_H

namespace FlexFlow {

std::optional<PCGBinarySPDecomposition>
    get_pcg_balanced_binary_sp_decomposition(
        ParallelComputationGraph const &);

} // namespace FlexFlow

#endif
