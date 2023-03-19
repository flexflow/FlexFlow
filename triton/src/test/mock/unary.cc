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

#include "operators/unary.h"

using namespace Legion;

namespace triton {
namespace backend {
namespace legion {

UnaryArgs::UnaryArgs() {}

UnaryOperator::UnaryOperator(LegionModelState *model,
                             LayerStrategy const *strategy,
                             OperatorType type,
                             void const *scalar,
                             DataType scalar_type,
                             bool inplace,
                             char const *name)
    : Operator(model,
               strategy,
               type,
               name,
               1 /*inputs*/,
               0 /*weights*/,
               1 /*outputs*/),
      scalar_type(scalar_type), inplace(inplace) {}

UnaryOperator::~UnaryOperator() {}

void UnaryOperator::Configure(Tensor *input, Tensor *output) {
  assert(input != nullptr);
  assert(output != nullptr);
  assert(input->type == scalar_type);
  assert((op_type == OP_CAST) || (input->type == output->type));
  assert(!inplace || (input == output));
  // Make sure that they have the same bounds
  assert(input->bounds.size() == output->bounds.size());
  for (unsigned idx = 0; idx < input->bounds.size(); idx++) {
    assert(input->bounds[idx] == output->bounds[idx]);
  }

  // Hack so that we can access the tensors in the tests
  auto vec_ptr = reinterpret_cast<std::vector<Tensor *> *>(model);
  vec_ptr->emplace_back(input);
  vec_ptr->emplace_back(output);
}

void UnaryOperator::Load(Realm::Processor processor) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void UnaryOperator::initialize(LegionModelInstance *instance,
                               unsigned const instance_index,
                               Legion::Runtime *runtime,
                               Legion::Context ctx,
                               Legion::MapperID mapper) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void UnaryOperator::forward(LegionModelInstance *instance,
                            unsigned const instance_index,
                            Legion::Runtime *runtime,
                            Legion::Context ctx,
                            Legion::MapperID mapper) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void UnaryOperator::finalize(LegionModelInstance *instance,
                             unsigned const instance_index,
                             Legion::Runtime *runtime,
                             Legion::Context ctx,
                             Legion::MapperID mapper) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void UnaryOperator::Free(Realm::Processor processor) {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void UnaryOperator::PreregisterTaskVariants() {
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

} // namespace legion
} // namespace backend
} // namespace triton
