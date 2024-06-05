// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/computation_graph/layer_added_result.struct.toml
/* proj-data
{
  "generated_from": "15bf9d73ef934599c9b11807d86ae5d4"
}
*/

#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_LAYER_ADDED_RESULT_DTG_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_LAYER_ADDED_RESULT_DTG_H

#include "fmt/format.h"
#include "pcg/layer_guid_t.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct LayerAddedResult {
  LayerAddedResult() = delete;
  LayerAddedResult(::FlexFlow::layer_guid_t const &layer,
                   std::vector<::FlexFlow::tensor_guid_t> const &outputs);

  bool operator==(LayerAddedResult const &) const;
  bool operator!=(LayerAddedResult const &) const;
  ::FlexFlow::layer_guid_t layer;
  std::vector<::FlexFlow::tensor_guid_t> outputs;
};
} // namespace FlexFlow

namespace FlexFlow {
std::string format_as(LayerAddedResult const &);
std::ostream &operator<<(std::ostream &, LayerAddedResult const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_LAYER_ADDED_RESULT_DTG_H
