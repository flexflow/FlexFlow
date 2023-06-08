/* Copyright 2022 NVIDIA CORPORATION
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "operators/softmax.h"

using namespace Legion;

namespace triton {
namespace backend {
namespace legion {

Softmax::Softmax(LegionModelState *model,
                 LayerStrategy const *strategy,
                 unsigned dim,
                 char const *name)
    : Operator(model, strategy, OperatorType::OP_SOFTMAX, name, 1, 0, 1),
      dim(dim) {}

void Softmax::Configure(Tensor *input, Tensor *output) {
  assert(input != nullptr);
  assert(output != nullptr);
  assert(input->type == output->type);
  // Make sure that they have the same bounds
  assert(input->bounds.size() == output->bounds.size());
  for (unsigned idx = 0; idx < input->bounds.size(); idx++) {
    assert(input->bounds[idx] == output->bounds[idx]);
  }
  inputs.push_back(input);
  outputs.push_back(output);

  // Hack so that we can access the tensors in the tests
  auto vec_ptr = reinterpret_cast<std::vector<Tensor *> *>(model);
  vec_ptr->emplace_back(input);
  vec_ptr->emplace_back(output);
}

Domain Softmax::GetBounds(Processor proc) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void Softmax::Load(Processor proc) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void Softmax::initialize(LegionModelInstance *instance,
                         unsigned const instance_index,
                         Runtime *runtime,
                         Context ctx,
                         MapperID mapper) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void Softmax::forward(LegionModelInstance *instance,
                      unsigned const instance_index,
                      Runtime *runtime,
                      Context ctx,
                      MapperID mapper) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void Softmax::finalize(LegionModelInstance *instance,
                       unsigned const instance_index,
                       Runtime *runtime,
                       Context ctx,
                       MapperID mapper) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void Softmax::Free(Processor proc) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

/*static*/ void Softmax::PreregisterTaskVariants(void) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

/*static*/ void Softmax::forward_cpu(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

SoftmaxArgs::SoftmaxArgs(void) {}

#ifdef LEGION_USE_CUDA
/*static*/ void Softmax::forward_gpu(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
#endif

} // namespace legion
} // namespace backend
} // namespace triton
