#include "flexflow/flexflow.h"
#include <stdbool.h>

int top_level_task_impl(int argc, char **argv) {
  flexflow_config_t config;
  CHECK_FLEXFLOW(flexflow_config_parse_argv(&argc, argv, true, &config));
  CHECK_FLEXFLOW(flexflow_set_config(config));

  flexflow_model_config_t model_config;
  CHECK_FLEXFLOW(flexflow_model_config_parse_argv(&argc, argv, true, &config));

  flexflow_computation_graph_t cg;
  CHECK_FLEXFLOW(flexflow_computation_graph_create(&cg));
  CHECK_FLEXFLOW(flexflow_set_model_config(cg, model_config));

  flexflow_tensor_t tensor;
  int dims[] = {5, 10, 16};
  CHECK_FLEXFLOW(flexflow_computation_graph_create_tensor(
      cg, 3, dims, FLEXFLOW_DATATYPE_FLOAT, true, &tensor));
}
