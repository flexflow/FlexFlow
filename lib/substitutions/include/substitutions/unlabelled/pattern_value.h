#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_VALUE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_VALUE_H

#include "substitutions/unlabelled/pattern_value.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

OpenDataflowValue raw_dataflow_value_from_pattern_value(PatternValue const &);

} // namespace FlexFlow

#endif
