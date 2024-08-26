#include "utils/graph/open_dataflow_graph/algorithms/generate_new_input_id_permutation.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"
#include "utils/graph/open_dataflow_graph/dataflow_graph_input_source.h"

namespace FlexFlow {

bidict<NewDataflowGraphInput, DataflowGraphInput>
    generate_new_input_id_permutation(OpenDataflowGraphView const &g) {
  DataflowGraphInputSource input_source;
  return generate_bidict(get_inputs(g),
                         [&](DataflowGraphInput const &) {
                           return NewDataflowGraphInput{
                               input_source.new_dataflow_graph_input()};
                         })
      .reversed();
}

} // namespace FlexFlow
