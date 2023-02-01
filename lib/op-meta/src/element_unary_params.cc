#include "op-meta/ops/element_unary_params.h"

namespace FlexFlow {

typename ElementUnaryParams::AsConstTuple ElementUnaryParams::as_tuple() const {
  return {this->op, this->inplace, this->scalar};
}

bool operator==(ElementUnaryParams const &lhs, ElementUnaryParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(ElementUnaryParams const &lhs, ElementUnaryParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

}
