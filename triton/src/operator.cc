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

#include "operator.h"
#include "operators/binary.h"
#include "operators/concat.h"
#include "operators/conv2d.h"
#include "operators/matmul.h"
#include "operators/reshape.h"
#include "operators/softmax.h"
#include "operators/unary.h"
#include "tensor.h"

namespace triton { namespace backend { namespace legion {

Operator::Operator(
    LegionModelState* m, const LayerStrategy* s, OperatorType t,
    const char* name, unsigned in, unsigned wts, unsigned out)
    : op_type(t), op_name(name), model(m), strategy(s), num_inputs(in),
      num_weights(wts), num_outputs(out)
{
}

Operator::~Operator(void)
{
  // Delete all the weight and output tensors
  for (auto wts : weights) delete wts;
  for (auto tensor : outputs) delete tensor;
}

/*static*/ void
Operator::PreregisterTaskVariants(void)
{
  BinaryOperator::PreregisterTaskVariants();
  Concat::PreregisterTaskVariants();
  Conv2D::PreregisterTaskVariants();
  MatMul::PreregisterTaskVariants();
  Reshape::PreregisterTaskVariants();
  Softmax::PreregisterTaskVariants();
  UnaryOperator::PreregisterTaskVariants();
}

}}}  // namespace triton::backend::legion
