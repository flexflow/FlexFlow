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

#include "operators/conv2d.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

Conv2D::Conv2D(
    LegionModelState* model, const LayerStrategy* strategy, size_t inChannels,
    size_t outChannels, size_t kernelH, size_t kernelW, size_t strideH,
    size_t strideW, size_t paddingH, size_t paddingW, ActivationMode act,
    size_t gps, bool bias, const char* name)
    : Operator(
          model, strategy, OP_CONV2D, name, 1 /*inputs*/,
          bias ? 2 : 1 /*weights*/, 1 /*outputs*/),
      activation(act), in_channels(inChannels), out_channels(outChannels),
      kernel_h(kernelH), kernel_w(kernelW), stride_h(strideH),
      stride_w(strideW), padding_h(paddingH), padding_w(paddingW), groups(gps),
      use_bias(bias)
{
}

Conv2D::~Conv2D() {}

void
Conv2D::Configure(Tensor* input, Weights* wts, Tensor* output, Weights* bias)
{
  assert(input != nullptr);
  assert(in_channels == input->bounds[1]);
  assert(wts != nullptr);
  assert(output != nullptr);
  if (use_bias)
    assert(bias != nullptr);
  else
    assert(bias == nullptr);
  inputs.push_back(input);
  outputs.push_back(output);
  weights.push_back(wts);
  if (use_bias)
    weights.push_back(bias);
  // Hack so that we can access the tensors in the tests
  auto vec_ptr = reinterpret_cast<std::vector<Tensor*>*>(model);
  vec_ptr->emplace_back(input);
  vec_ptr->emplace_back(wts);
  if (use_bias) {
    vec_ptr->emplace_back(bias);
  }
  vec_ptr->emplace_back(output);
}

Rect<4>
Conv2D::GetWeightBounds(Realm::Processor proc)
{
  if ((weights.size() < 1) || (weights.size() > 2)) {
    throw std::invalid_argument("Weight is not configured for Conv2D operator");
  }
  // Splitting the H, W without actually looking at the
  // patitioning in LayerStrategy, but the positioning in terms of
  // global processor, which is what we can control in mock LayerStrategy.
  size_t h_stride =
      (weights[0]->bounds[2] - (strategy->global_processors.size() - 1)) /
      strategy->global_processors.size();
  size_t w_stride =
      (weights[0]->bounds[3] - (strategy->global_processors.size() - 1)) /
      strategy->global_processors.size();
  DomainPoint lo, hi;
  lo.dim = 4;
  lo[0] = 0;
  lo[1] = 0;
  hi.dim = 4;
  hi[0] = weights[0]->bounds[0] - 1;
  hi[1] = weights[0]->bounds[1] - 1;
  for (size_t idx = 0; idx < strategy->global_processors.size(); ++idx) {
    if (proc == strategy->global_processors[idx]) {
      lo[2] = h_stride * idx;
      hi[2] = h_stride * (idx + 1) - 1;
      if (hi[2] > (weights[0]->bounds[2] - 1)) {
        hi[2] = (weights[0]->bounds[2] - 1);
      }

      lo[3] = w_stride * idx;
      hi[3] = w_stride * (idx + 1) - 1;
      if (hi[3] > (weights[0]->bounds[3] - 1)) {
        hi[3] = (weights[0]->bounds[3] - 1);
      }
    }
  }
  return Rect<4>(lo, hi);
}

Rect<1>
Conv2D::GetBiasBounds(Realm::Processor proc)
{
  if (weights.size() != 2) {
    throw std::invalid_argument("Bias is not configured for Conv2D operator");
  }
  // Always return the whole bias bound
  DomainPoint lo, hi;
  lo.dim = 1;
  lo[0] = 0;
  hi.dim = 1;
  hi[0] = weights[1]->bounds[0] - 1;
  return Rect<1>(lo, hi);
}

void
Conv2D::Load(Realm::Processor processor)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void
Conv2D::initialize(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void
Conv2D::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void
Conv2D::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void
Conv2D::Free(Realm::Processor processor)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
Conv2D::PreregisterTaskVariants()
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

}}}  // namespace triton::backend::legion
