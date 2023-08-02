#ifndef _FLEXFLOW_COMPILER_INCLUDE_COMPILER_FFI_COMPILER_H
#define _FLEXFLOW_COMPILER_INCLUDE_COMPILER_FFI_COMPILER_H

#include "flexflow/pcg.h"
#include "flexflow/utils.h"
#include <stdio.h>

FLEXFLOW_FFI_BEGIN()


typedef enum {
  FLEXFLOW_SEARCH_ALGORITHM_DATA_PARALLEL
} flexflow_search_algorithm_t;

FF_NEW_OPAQUE_TYPE(flexflow_search_algorithm_config_t);
FF_NEW_OPAQUE_TYPE(flexflow_search_result_t);
FF_NEW_OPAQUE_TYPE(flexflow_cost_estimator_t);

// Error functions
typedef enum {
  FLEXFLOW_COMPILER_STATUS_OK,
  FLEXFLOW_COMPILER_ERROR_UNKNOWN
} flexflow_compiler_error_code_t;


FF_NEW_OPAQUE_TYPE(flexflow_compiler_error_t);

flexflow_error_t flexflow_compiler_error_wrap(flexflow_compiler_error_t);
flexflow_error_t flexflow_compiler_error_unwrap(flexflow_error_t, 
                                                flexflow_compiler_error_t *);
flexflow_error_t flexflow_compiler_error_is_ok(flexflow_compiler_error_t, bool *);
flexflow_error_t flexflow_compiler_error_get_string(flexflow_compiler_error_t, char **message);
flexflow_error_t flexflow_compiler_error_get_error_code(flexflow_compiler_error_t, flexflow_compiler_error_code_t *out);
flexflow_error_t flexflow_compiler_error_destroy(flexflow_compiler_error_t);

//

flexflow_error_t
    flexflow_computation_graph_optimize(flexflow_computation_graph_t,
                                        flexflow_machine_specification_t,
                                        flexflow_search_algorithm_config_t,
                                        flexflow_search_result_t *out);

flexflow_error_t flexflow_search_result_get_parallel_computation_graph(
    flexflow_search_result_t, flexflow_parallel_computation_graph_t *out);

flexflow_error_t flexflow_search_result_get_parallel_tensor_for_tensor(
    flexflow_search_result_t,
    flexflow_tensor_t,
    flexflow_parallel_tensor_t *out);

FLEXFLOW_FFI_END()

#endif
