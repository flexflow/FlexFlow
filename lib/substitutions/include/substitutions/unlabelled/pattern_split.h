#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_SPLIT_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_SPLIT_H

#include "substitutions/unlabelled/edge_splits.dtg.h"
#include "substitutions/unlabelled/pattern_split.dtg.h"
#include "substitutions/unlabelled/pattern_split_result.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"

namespace FlexFlow {

PatternSplit find_even_split(UnlabelledGraphPattern const &);

// GraphSplit get_raw_split(PatternSplit const &);

// UnlabelledPatternEdgeSplits
//     get_edge_splits(UnlabelledGraphPattern const &pattern,
//                     PatternSplit const &split);
//
PatternSplitResult apply_split(UnlabelledGraphPattern const &,
                               PatternSplit const &);

} // namespace FlexFlow

#endif
