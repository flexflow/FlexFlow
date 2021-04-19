#include "ops/linear.h"
#include "legion/legion_utilities.h"

void Linear::serialize(Legion::Serializer& sez) const { 
  sez.serialize(this->out_channels); 
  sez.serialize(this->activation); 
  sez.serialize(this->use_bias); 
} 

/* static */
Node Linear::deserialize(FFModel &ff, Legion::Deserializer &dez, Tensor inputs[], int num_inputs) { 
  assert (num_inputs == 1); 
  int out_channels; 
  ActiMode activation; 
  bool use_bias; 
  dez.deserialize(out_channels); 
  dez.deserialize(activation); 
  dez.deserialize(use_bias); 
  return ff.get_or_create_linear_node(inputs[0], out_channels, activation, use_bias); 
} 
