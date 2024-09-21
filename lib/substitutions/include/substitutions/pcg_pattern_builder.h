#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_PCG_PATTERN_BUILDER_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_PCG_PATTERN_BUILDER_H

#include "substitutions/operator_pattern/operator_attribute_pattern.dtg.h"
#include "substitutions/pcg_pattern.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.dtg.h"
#include "substitutions/unlabelled/pattern_value.dtg.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph.h"

namespace FlexFlow {

struct PCGPatternBuilder {
  PCGPatternBuilder();

  PatternValue add_input();
  PatternValue add_input(TensorAttributePattern const &);

  std::vector<PatternValue>
      add_operator(OperatorAttributePattern const &,
                   std::vector<PatternValue> const &inputs,
                   std::vector<TensorAttributePattern> const &outputs);
  PatternValue add_operator(OperatorAttributePattern const &,
                            std::vector<PatternValue> const &inputs,
                            TensorAttributePattern const &output);

  PCGPattern get_pattern() const;

private:
  LabelledOpenDataflowGraph<OperatorAttributePattern, TensorAttributePattern> g;
};

} // namespace FlexFlow

#endif
