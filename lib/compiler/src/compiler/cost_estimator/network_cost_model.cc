#include "compiler/cost_estimator/network_cost_model.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

float estimate_communication_cost(
    MachineSpecification const &machine_spec,
    TensorSetMovement const &tensor_set_movement) {
  NOT_IMPLEMENTED(); // TODO @lockshaw
  // for (SingleTensorMovement const &single_tensor_movement :
  // tensor_set_movement.single_tensor_movements) {
  //   for
  // }
}

} // namespace FlexFlow
