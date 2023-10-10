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

const PEFTModelID PEFTModelID::NO_ID = PEFTModelID();

PEFTModelID::PEFTModelID() : id(0) {}

PEFTModelID::PEFTModelID(size_t _id) : id(_id) {
  assert(is_valid_id());
}

bool PEFTModelID::is_valid_id() const {
  return (id >= PEFT_MODEL_ID_FIRST_VALID && id <= PEFT_MODEL_ID_LAST_VALID);
}

bool operator==(PEFTModelID const &lhs, PEFTModelID const &rhs) {
  return lhs.id == rhs.id;
}

}; // namespace FlexFlow
