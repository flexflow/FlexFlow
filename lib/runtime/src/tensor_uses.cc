#include "tensor_uses.h"

namespace FlexFlow {

TensorUseDescription::TensorUseDescription(TensorUseType const &type,
                                           Layer const *layer,
                                           int idx)
    : type(type), layer(layer), idx(idx) {}

std::vector<TensorUseDescription> TensorUses::at(Tensor const &tensor) const {
  return this->at(tensor->tensor_guid);
}

std::vector<TensorUseDescription> TensorUses::at(TensorBase const *base) const {
  return this->at(base->tensor_guid);
}

std::vector<TensorUseDescription> TensorUses::at(size_t tensor_guid) const {
  return this->uses.at(tensor_guid);
}

void TensorUses::remove(Layer const &layer) {
  for (auto const &k : keys(this->uses)) {
    inplace_filter(this->uses.at(k), [&](TensorUseDescription const &d) {
      return d.layer->layer_guid == layer.layer_guid;
    });
  }
}

void TensorUses::update(Layer const &layer) {
  this->remove(layer);
  for (int idx = 0; idx < layer.outputs.size(); idx++) {
    Tensor output = layer.outputs.at(idx);
    this->uses[output->tensor_guid].push_back(
        {TensorUseType::OUTPUT, &layer, idx});
  }
  for (int idx = 0; idx < layer.weights.size(); idx++) {
    Tensor weight = layer.weights.at(idx);
    this->uses[weight->tensor_guid].push_back(
        {TensorUseType::WEIGHT, &layer, idx});
  }
  for (int idx = 0; idx < layer.inputs.size(); idx++) {
    Tensor input = layer.inputs.at(idx);
    this->uses[input->tensor_guid].push_back(
        {TensorUseType::INPUT, &layer, idx});
  }
}

} // namespace FlexFlow
