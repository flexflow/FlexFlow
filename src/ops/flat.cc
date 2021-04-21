#include "ops/flat.h"

void Flat::serialize(Legion::Serializer& sez) const {
  return; 
}

/*static*/
Node Flat::deserialize(FFModel& ff, Legion::Deserializer& dez, Tensor inputs[], int num_inputs) {
  assert (num_inputs == 1);
  return ff.get_or_create_flat_node(inputs[0]);
}

Op *Flat::materialize(FFModel& ff, Tensor inputs[], int num_inputs) const {
  assert (num_inputs == 1);
  return new Flat(ff, inputs[0], this->name);
}
