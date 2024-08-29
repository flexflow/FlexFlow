#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_SPLIT_SP_DECOMPOSITION_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_SPLIT_SP_DECOMPOSITION_H

#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"

namespace FlexFlow {

std::pair<SerialParallelDecomposition, SerialParallelDecomposition>
    split_sp_decomposition(SerialSplit const &serial);

std::pair<SerialParallelDecomposition, SerialParallelDecomposition>
    split_sp_decomposition(ParallelSplit const &parallel);

}

#endif