#include "optimizations.h"

namespace FlexFlow {

void FFModel::optimize_unnecessary_gradient_calculations() {
  // If an operator's input is training data
  // No need to compute its gradients
  for (size_t l = 0; l < operators.size(); l++) {
    Op *op = operators[l];
    for (int i = 0; i < op->numInputs; i++) {
      assert(op->inputs[i]->owner_op != nullptr);
      if (op->inputs[i]->owner_op->op_type == OP_INPUT) {
        op->trainableInputs[i] = false;
      }
    }
  }
}

}
