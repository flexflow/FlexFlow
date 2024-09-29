#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_RESULT_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_RESULT_H

#include "compiler/machine_mapping/machine_mapping_result.dtg.h"

namespace FlexFlow {

MachineMappingResult get_infinity_machine_mapping_result();

void minimize_runtime(MachineMappingResult &m1, MachineMappingResult const &m2);

} // namespace FlexFlow

#endif
