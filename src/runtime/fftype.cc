#include "flexflow/fftype.h"
#include <cassert>

namespace FlexFlow {

LayerID::LayerID() : id(0) {}

LayerID::LayerID(size_t _id) : id(_id) {
  assert(is_valid_id());
}

bool LayerID::is_valid_id() const {
  return (id >= LAYER_GUID_FIRST_VALID && id <= LAYER_GUID_LAST_VALID);
}

bool operator==(LayerID const &lhs, LayerID const &rhs) {
  return lhs.id == rhs.id;
}

}; // namespace FlexFlow