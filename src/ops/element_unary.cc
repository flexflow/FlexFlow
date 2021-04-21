#include "ops/element_unary.h"
#include "legion/legion_utilities.h"

void ElementUnary::serialize(Legion::Serializer& sez) const {
  sez.serialize(this->op_type);
  sez.serialize(this->inplace);
  if (this->op_type == OP_SCALAR_MULTIPLY) {
    sez.serialize(scalar);
  }
}

/*static*/
Node ElementUnary::deserialize(FFModel& ff, Legion::Deserializer& dez, Tensor inputs[], int num_inputs) {
  assert (num_inputs == 1);
  OperatorType op_type;
  float scalar;
  bool inplace;
  dez.deserialize(op_type);
  dez.deserialize(inplace);
  if (op_type == OP_SCALAR_MULTIPLY) {
    dez.deserialize(scalar);
  }

  return ff.get_or_create_element_unary_node(inputs[0], op_type, inplace, scalar);
}

Op *ElementUnary::materialize(FFModel& ff, Tensor inputs[], int num_inputs) const {
  assert (num_inputs == 1);
  return new ElementUnary(ff, this->op_type, inputs[0], this->inplace, this->name, this->scalar);
}
