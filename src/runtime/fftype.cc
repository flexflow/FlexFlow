#include "flexflow/fftype.h"
#include "flexflow/config.h"
#include <cassert>

namespace FlexFlow {

const LayerID LayerID::NO_ID = LayerID();

LayerID::LayerID()
    : id(0), transformer_layer_id(MAX_NUM_TRANSFORMER_LAYERS), model_id(0) {}

LayerID::LayerID(size_t _id, size_t _transformer_layer_id, size_t _model_id)
    : id(_id), transformer_layer_id(_transformer_layer_id),
      model_id(_model_id) {
  assert(is_valid_id());
}

bool LayerID::is_valid_id() const {
  return (id >= LAYER_GUID_FIRST_VALID && id <= LAYER_GUID_LAST_VALID &&
          transformer_layer_id >= 0 &&
          transformer_layer_id < MAX_NUM_TRANSFORMER_LAYERS && model_id >= 0);
}

bool operator==(LayerID const &lhs, LayerID const &rhs) {
  // id should be sufficient to distinguish different layers
  if (lhs.id == rhs.id) {
    assert(lhs.transformer_layer_id == rhs.transformer_layer_id);
    assert(lhs.model_id == rhs.model_id);
  }
  return lhs.id == rhs.id;
}

}; // namespace FlexFlow
