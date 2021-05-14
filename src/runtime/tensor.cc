#include "tensor.h"

bool TensorShape::operator==(const TensorShape& other) const {
  if (this->num_dims != other.num_dims) {
    return false;
  }

  for (int i = 0; i < this->num_dims; i++) { 
    if (this->dims[i].size != other.dims[i].size) {
      return false;
    }

    if (this->dims[i].degree != other.dims[i].degree) { 
      return false;
    }
  }

  return true;
}
