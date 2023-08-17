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

#include "operators/matmul.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

MatMul::MatMul(
    LegionModelState* model, const LayerStrategy* strategy, const char* name)
    : Operator(model, strategy, OperatorType::OP_MATMUL, name, 2, 0, 1)
{
}

template <unsigned DIM>
void
MatMul::compute_in1_parameters(Tensor* in1, Tensor* out)
{
  Rect<DIM> extent, colors;
  Transform<DIM, DIM> transform;
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++) transform[i][j] = 0;
  assert(out->bounds.size() >= in1->bounds.size());
  size_t dimoff = out->bounds.size() - in1->bounds.size();
  for (int i = 0; i < DIM; i++) {
    extent.lo[i] = 0;
    colors.lo[i] = 0;
    if (i == (DIM - 1)) {
      /* need the whole dimension */
      extent.hi[i] = in1->bounds[i] - 1; /*inclusive*/
      colors.hi[i] = 0;
    } else if (in1->bounds[i] == 1) {
      extent.hi[i] = 0;
      colors.hi[i] = 0;
    } else {
      size_t pieces = strategy->dim[dimoff + 1];
      size_t chunks = (in1->bounds[i] + pieces - 1) / pieces;
      extent.hi[i] = chunks - 1; /*inclusive*/
      colors.hi[i] = pieces - 1; /*inclusive*/
    }
  }
  in1_transform = transform;
  in1_extent = extent;
  in1_colors = colors;
}

template <unsigned DIM>
void
MatMul::compute_in2_parameters(Tensor* in2, Tensor* out)
{
  Rect<DIM> extent, colors;
  Transform<DIM, DIM> transform;
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++) transform[i][j] = 0;
  assert(out->bounds.size() >= in2->bounds.size());
  size_t dimoff = out->bounds.size() - in2->bounds.size();
  for (int i = 0; i < DIM; i++) {
    extent.lo[i] = 0;
    colors.lo[i] = 0;
    if (i == (DIM - 2)) {
      /* need the whole dimension */
      extent.hi[i] = in2->bounds[i] - 1; /*inclusive*/
      colors.hi[i] = 0;
    } else if (in2->bounds[i] == 1) {
      extent.hi[i] = 0;
      colors.hi[i] = 0;
    } else {
      size_t pieces = strategy->dim[dimoff + i];
      size_t chunks = (in2->bounds[i] + pieces - 1) / pieces;
      extent.hi[i] = chunks - 1; /*inclusive*/
      colors.hi[i] = pieces - 1; /*inclusive*/
    }
  }
  in2_transform = transform;
  in2_extent = extent;
  in2_colors = colors;
}

void
MatMul::Configure(Tensor* in1, Tensor* in2, Tensor* out)
{
  assert(in1 != nullptr);
  assert(in2 != nullptr);
  assert(out != nullptr);
  inputs.push_back(in1);
  inputs.push_back(in2);
  outputs.push_back(out);

  if ((in1->bounds.size() == 1) && (in2->bounds.size() == 1)) {
    fprintf(stderr, "TODO: support for dot-product in matmul operator");
    abort();
  } else if (in1->bounds.size() == 1) {
    const size_t in2_dim = in2->bounds.size();
    const size_t out_dim = out->bounds.size();
    assert(in2_dim >= 2);
    assert(out_dim >= 1);
    const size_t n = out->bounds[out_dim - 1];
    const size_t k = in1->bounds[0];
    assert(in2->bounds[in2_dim - 2] == k);
    assert(in2->bounds[in2_dim - 1] == n);
    // make sure all the other dimensions align or broadcast
    unsigned in2_broadcasts = 0;
    for (unsigned off = 3; off <= out_dim; off++) {
      const size_t out_size = out->bounds[out_dim - off];
      if (off <= in2_dim) {
        const size_t size = in2->bounds[in2_dim - off];
        assert((size == 1) || (size == out_size));
        if (size == 1)
          in2_broadcasts |= (1 << (off - 3));
      }
    }

    Rect<1> extent, colors;
    Transform<1, 1> transform;
    transform[0][0] = 0;
    extent.lo[0] = 0;
    extent.hi[0] = in1->bounds[0] - 1;  // inclusive
    colors.lo[0] = 0;
    colors.hi[0] = 0;
    in1_transform = transform;
    in1_extent = extent;
    in1_colors = colors;

    switch (in2->bounds.size()) {
#define DIMFUNC(DIM)                       \
  case DIM: {                              \
    compute_in2_parameters<DIM>(in2, out); \
    break;                                 \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
  } else if (in2->bounds.size() == 1) {
    const size_t in1_dim = in1->bounds.size();
    const size_t out_dim = out->bounds.size();
    assert(in1_dim >= 2);
    const size_t m = (out_dim > 1) ? out->bounds[out_dim - 2] : 1;
    assert(out->bounds[out_dim - 1] == 1);
    assert(in1->bounds[in1_dim - 2] == m);
    const size_t k = in1->bounds[in1_dim - 1];
    assert(in2->bounds[in2->bounds[0]] == k);
    // make sure all the other dimensions align or broadcast
    unsigned in1_broadcasts = 0;
    for (unsigned off = 3; off <= out_dim; off++) {
      const size_t out_size = out->bounds[out_dim - off];
      if (off <= in1_dim) {
        const size_t size = in1->bounds[in1_dim - off];
        assert((size == 1) || (size == out_size));
        if (size == 1)
          in1_broadcasts |= (1 << (off - 3));
      }
    }

    switch (in1->bounds.size()) {
#define DIMFUNC(DIM)                       \
  case DIM: {                              \
    compute_in1_parameters<DIM>(in1, out); \
    break;                                 \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }

    Rect<1> extent, colors;
    Transform<1, 1> transform;
    transform[0][0] = 0;
    extent.lo[0] = 0;
    extent.hi[0] = in2->bounds[0] - 1;  // inclusive
    colors.lo[0] = 0;
    colors.hi[0] = 0;
    in2_transform = transform;
    in2_extent = extent;
    in2_colors = colors;

  } else {
    // all tensors have at least two dimensions
    const size_t in1_dim = in1->bounds.size();
    const size_t in2_dim = in2->bounds.size();
    const size_t out_dim = out->bounds.size();
    assert(in1_dim >= 2);
    assert(in2_dim >= 2);
    assert(out_dim >= 2);
    const size_t m = out->bounds[out_dim - 2];
    const size_t n = out->bounds[out_dim - 1];
    assert(in1->bounds[in1_dim - 2] == m);
    const size_t k = in1->bounds[in1_dim - 1];
    assert(in2->bounds[in2_dim - 2] == k);
    assert(in2->bounds[in2_dim - 1] == n);
    // make sure all the other dimensions align or can broadcast
    unsigned in1_broadcasts = 0, in2_broadcasts = 0;
    for (unsigned off = 3; off <= out_dim; off++) {
      const size_t out_size = out->bounds[out_dim - off];
      if (off <= in1_dim) {
        const size_t size = in1->bounds[in1_dim - off];
        assert((size == 1) || (size == out_size));
        if (size == 1)
          in1_broadcasts |= (1 << (off - 3));
      }
      if (off <= in2_dim) {
        const size_t size = in2->bounds[in2_dim - off];
        assert((size == 1) || (size == out_size));
        if (size == 1)
          in2_broadcasts |= (1 << (off - 3));
      }
    }

    // Finally fill in the input transforms, extents, and colors for the inputs
    switch (in1->bounds.size()) {
#define DIMFUNC(DIM)                       \
  case DIM: {                              \
    compute_in1_parameters<DIM>(in1, out); \
    break;                                 \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }

    switch (in2->bounds.size()) {
#define DIMFUNC(DIM)                       \
  case DIM: {                              \
    compute_in2_parameters<DIM>(in2, out); \
    break;                                 \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
  }

  // Hack so that we can access the tensors in the tests
  auto vec_ptr = reinterpret_cast<std::vector<Tensor*>*>(model);
  vec_ptr->emplace_back(in1);
  vec_ptr->emplace_back(in2);
  vec_ptr->emplace_back(out);
}

Domain
MatMul::GetIn1Bounds(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

Domain
MatMul::GetIn2Bounds(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

Domain
MatMul::GetOutBounds(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
MatMul::Load(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
MatMul::initialize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
MatMul::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
MatMul::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
MatMul::Free(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

/*static instantiations*/
MatMul::FunctorTable MatMul::in1_functors;
MatMul::FunctorTable MatMul::in2_functors;

/*static*/ void
MatMul::PreregisterTaskVariants(void)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

/*static*/ void
MatMul::forward_cpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

MatMulArgs::MatMulArgs(void) {}

#ifdef LEGION_USE_CUDA
/*static*/ void
MatMul::forward_gpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
#endif

}}}  // namespace triton::backend::legion
