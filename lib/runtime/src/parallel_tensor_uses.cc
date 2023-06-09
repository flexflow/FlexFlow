#include "parallel_tensor_uses.h"
#include "operator.h"

namespace FlexFlow {

std::vector<ParallelTensorUseDescription>
    ParallelTensorUses::at(ParallelTensor const &p_tensor) const {
  return this->at(p_tensor->parallel_tensor_guid);
}

std::vector<ParallelTensorUseDescription>
    ParallelTensorUses::at(ParallelTensorBase const *ptr) const {
  return this->at(ptr->parallel_tensor_guid);
}

std::vector<ParallelTensorUseDescription>
    ParallelTensorUses::at(size_t parallel_tensor_guid) const {
  return this->uses.at(parallel_tensor_guid);
}

Op const *ParallelTensorUses::get_owner(ParallelTensor const &tensor) const {
  Op const *result = nullptr;
  for (ParallelTensorUseDescription const &d : this->at(tensor)) {
    if (d.type == TensorUseType::OUTPUT) {
      assert(result == nullptr);
      result = d.op;
    }
  }
  assert(result != nullptr);
  return result;
}

void ParallelTensorUses::remove(Op const &op) {
  for (auto const &k : keys(this->uses)) {
    inplace_filter(this->uses.at(k),
                   [&](ParallelTensorUseDescription const &d) {
                     return d.op->op_guid == op.op_guid;
                   });
  }
}

void ParallelTensorUses::update(Op const &op) {
  this->remove(op);
  for (int idx = 0; idx < op.outputs.size(); idx++) {
    ParallelTensor output = op.outputs.at(idx);
    this->uses[output->parallel_tensor_guid].push_back(
        {TensorUseType::OUTPUT, &op, idx});
  }
  for (int idx = 0; idx < op.weights.size(); idx++) {
    ParallelTensor weight = op.weights.at(idx);
    this->uses[weight->parallel_tensor_guid].push_back(
        {TensorUseType::WEIGHT, &op, idx});
  }
  for (int idx = 0; idx < op.inputs.size(); idx++) {
    ParallelTensor input = op.inputs.at(idx);
    this->uses[input->parallel_tensor_guid].push_back(
        {TensorUseType::INPUT, &op, idx});
  }
}

} // namespace FlexFlow
