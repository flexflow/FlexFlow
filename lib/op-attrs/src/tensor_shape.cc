#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

size_t TensorShape::size_t at(ff_dim_t dim) const {
    return dims.at(dim);
}

size_t TensorShape::operator[](ff_dim_t dim ) const {
    return at(dim);
}

}
