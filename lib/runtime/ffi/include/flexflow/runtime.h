#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_FFI_RUNTIME_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_FFI_RUNTIME_H

#include "flexflow/compiler.h"
#include "flexflow/op-attrs.h"
#include "flexflow/pcg.h"
#include "flexflow/utils.h"
#include <stdbool.h>
#include <stdint.h>

FLEXFLOW_FFI_BEGIN()

FF_NEW_OPAQUE_TYPE(flexflow_config_t);
FF_NEW_OPAQUE_TYPE(flexflow_model_config_t);
FF_NEW_OPAQUE_TYPE(flexflow_model_training_instance_t);
FF_NEW_OPAQUE_TYPE(flexflow_void_future_t);

typedef enum {
  FLEXFLOW_RUNTIME_STATUS_OK,
  FLEXFLOW_RUNTIME_ERROR_UNKNOWN,
  FLEXFLOW_RUNTIME_ERROR_DYNAMIC_ALLOCATION_FAILED,
  FLEXFLOW_RUNTIME_ERROR_UNEXPECTED_EMPTY_HANDLE,
} flexflow_runtime_error_t;

typedef enum {
  FLEXFLOW_METRIC_ACCURACY,
  FLEXFLOW_METRIC_CATEGORICAL_CROSSENTROPY,
  FLEXFLOW_METRIC_SPARSE_CATEGORICAL_CROSSENTROPY,
  FLEXFLOW_METRIC_MEAN_SQUARED_ERROR,
  FLEXFLOW_METRIC_ROOT_MEAN_SQUARED_ERROR,
  FLEXFLOW_METRIC_MEAN_ABSOLUTE_ERROR,
} flexflow_metric_t;

typedef enum {
  FLEXFLOW_LOSS_FUNCTION_CATEGORICAL_CROSSENTROPY,
  FLEXFLOW_LOSS_FUNCTION_SPARSE_CATEGORICAL_CROSSENTROPY,
  FLEXFLOW_LOSS_FUNCTION_MEAN_SQUARED_ERROR_AVG_REDUCE,
  FLEXFLOW_LOSS_FUNCTION_MEAN_SQUARED_ERROR_SUM_REDUCE,
  FLEXFLOW_LOSS_FUNCTION_IDENTITY
} flexflow_loss_function_t;

typedef enum {
  FLEXFLOW_COMPUTATION_MODE_TRAINING,
  FLEXFLOW_COMPUTATION_MODE_INFERENCE,
} flexflow_computation_mode_t;

char *flexflow_runtime_get_error_string(flexflow_runtime_error_t);

flexflow_runtime_error_t flexflow_void_future_wait(flexflow_void_future_t);
flexflow_runtime_error_t flexflow_void_future_destroy(flexflow_void_future_t);

flexflow_runtime_error_t flexflow_config_parse_argv(int *argc,
                                                    char **argv,
                                                    bool remove_used,
                                                    flexflow_config_t *out);
flexflow_runtime_error_t flexflow_set_config(flexflow_config_t);
flexflow_runtime_error_t flexflow_get_config(flexflow_config_t *);

flexflow_runtime_error_t flexflow_model_config_parse_argv(
    int *argc, char **argv, bool remove_used, flexflow_model_config_t *out);

flexflow_runtime_error_t
    flexflow_computation_graph_set_model_config(flexflow_computation_graph_t,
                                                flexflow_model_config_t);
flexflow_runtime_error_t
    flexflow_computation_graph_get_model_config(flexflow_computation_graph_t,
                                                flexflow_model_config_t *out);
flexflow_runtime_error_t flexflow_computation_graph_compile(
    flexflow_computation_graph_t,
    flexflow_optimizer_t,
    flexflow_model_compilation_result_t *out);

flexflow_runtime_error_t flexflow_model_compilation_result_get_pcg(
    flexflow_model_compilation_result_t,
    flexflow_parallel_computation_graph_t *out);
flexflow_runtime_error_t
    flexflow_model_compilation_result_get_parallel_tensor_for_tensor(
        flexflow_model_compilation_result_t,
        flexflow_tensor_t,
        flexflow_parallel_tensor_t *);

flexflow_runtime_error_t
    flexflow_start_training(flexflow_model_compilation_result_t,
                            flexflow_model_training_instance_t *);
flexflow_runtime_error_t
    flexflow_model_training_instance_forward(flexflow_model_training_instance_t,
                                             flexflow_void_future_t *out);
flexflow_runtime_error_t flexflow_model_training_instance_backward(
    flexflow_model_training_instance_t);
flexflow_runtime_error_t
    flexflow_stop_training(flexflow_model_training_instance_t);

flexflow_runtime_error_t
    flexflow_get_tensor_float(flexflow_model_training_instance_t,
                              flexflow_tensor_t,
                              float *data,
                              bool get_gradients);
flexflow_runtime_error_t
    flexflow_get_tensor_double(flexflow_model_training_instance_t,
                               flexflow_tensor_t,
                               float *data,
                               bool get_gradients);
flexflow_runtime_error_t
    flexflow_get_tensor_int32(flexflow_model_training_instance_t,
                              flexflow_tensor_t,
                              int32_t *data,
                              bool get_gradients);
flexflow_runtime_error_t
    flexflow_get_tensor_int64(flexflow_model_training_instance_t,
                              flexflow_tensor_t,
                              int64_t *data,
                              bool get_gradients);

flexflow_runtime_error_t flexflow_set_tensor_int(
    flexflow_model_training_instance_t, flexflow_tensor_t, int32_t *data);

FLEXFLOW_FFI_END()

#endif
