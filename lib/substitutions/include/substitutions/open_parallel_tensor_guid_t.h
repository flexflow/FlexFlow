#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPEN_PARALLEL_TENSOR_GUID_T_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPEN_PARALLEL_TENSOR_GUID_T_H

#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "substitutions/open_parallel_tensor_guid_t.dtg.h"
#include "substitutions/input_parallel_tensor_guid_t.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

open_parallel_tensor_guid_t open_parallel_tensor_guid_from_closed(parallel_tensor_guid_t);
open_parallel_tensor_guid_t open_parallel_tensor_guid_from_input(input_parallel_tensor_guid_t);

template <typename F, typename Ret = std::invoke_result_t<F, parallel_tensor_guid_t>>
Ret visit_open_parallel_tensor_guid(open_parallel_tensor_guid_t t, F f) {
  return t.raw_open_dataflow_value.visit<Ret>(overload {
    [&](DataflowOutput const &o) {
      return f(parallel_tensor_guid_t{o});
    },
    [&](DataflowGraphInput const &i) {
      return f(input_parallel_tensor_guid_t{i});
    },
  });
}


} // namespace FlexFlow

#endif
