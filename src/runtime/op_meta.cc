#include "flexflow/op_meta.h"
#include "flexflow/operator.h"

namespace FlexFlow {

OpMeta::OpMeta(FFHandler _handle) : handle(_handle), profiling(false) {
  for (int i = 0; i < MAX_NUM_INPUTS; i++) {
    trainableInputs[i] = true;
  }
  for (int i = 0; i < MAX_NUM_INPUTS; i++) {
    input_type[i] = DT_NONE;
  }
  for (int i = 0; i < MAX_NUM_WEIGHTS; i++) {
    weight_type[i] = DT_NONE;
  }
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    output_type[i] = DT_NONE;
  }
}

OpMeta::OpMeta(FFHandler _handle, Op const *op) : OpMeta(_handle) {
  for (int i = 0; i < op->numInputs; i++) {
    input_type[i] = op->inputs[i]->data_type;
  }
  for (int i = 0; i < op->numWeights; i++) {
    weight_type[i] = op->weights[i]->data_type;
  }
  for (int i = 0; i < op->numOutputs; i++) {
    output_type[i] = op->outputs[i]->data_type;
  }
}

} // namespace FlexFlow
