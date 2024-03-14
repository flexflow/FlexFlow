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

#ifndef __LEGION_TRITON_TYPES_H__
#define __LEGION_TRITON_TYPES_H__

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace triton { namespace backend { namespace legion {

enum DataType {
  DT_HALF,
  DT_FLOAT,
  DT_DOUBLE,
  DT_INT8,
  DT_INT16,
  DT_INT32,
  DT_INT64,
  DT_UINT8,
  DT_UINT16,
  DT_UINT32,
  DT_UINT64,
  DT_BOOLEAN,
  DT_NONE,
};

enum FieldName {
  FID_DATA,
};

static inline size_t
sizeof_datatype(DataType dt)
{
  assert(dt < DT_NONE);
  static const size_t sizes[DT_NONE] = {
      2,  // hard-code this since it's hard to express sizeof(__half)
      sizeof(float),
      sizeof(double),
      sizeof(int8_t),
      sizeof(int16_t),
      sizeof(int32_t),
      sizeof(int64_t),
      sizeof(uint8_t),
      sizeof(uint16_t),
      sizeof(uint32_t),
      sizeof(uint64_t),
      sizeof(bool),
  };
  return sizes[dt];
}

enum ActivationMode {
  AC_MODE_NONE,
  AC_MODE_RELU,
  AC_MODE_SIGMOID,
  AC_MODE_TANH,
  AC_MODE_GELU,
};

enum PoolType {
  POOL_MAX,
  POOL_AVG,
};

enum OperatorType {
  OP_INPUT,
  OP_WEIGHT,
  OP_NOOP,
  OP_CONV2D,
  OP_DROPOUT,
  OP_LINEAR,
  OP_BATCHMATMUL,
  OP_POOL2D,
  OP_SCALAR_ADD,
  OP_SCALAR_SUB,
  OP_SCALAR_MULTIPLY,
  OP_SCALAR_TRUE_DIV,
  OP_RELU,
  OP_IDENTITY,
  OP_SIGMOID,
  OP_TANH,
  OP_ELU,
  OP_FLAT,
  OP_SOFTMAX,
  OP_BATCHNORM,
  OP_CONCAT,
  OP_SPLIT,
  OP_EMBEDDING,
  OP_GROUP_BY,
  OP_AGGREGATE,
  // OP_ELEMENTWISE,
  OP_RESHAPE,
  OP_REVERSE,
  OP_TRANSPOSE,
  OP_EW_ADD,
  OP_EW_MUL,
  OP_MATMUL,
  OP_MUL,
  OP_ENLARGE,
  OP_MERGE_GCONV,
  OP_CONSTANT_IMM,
  OP_CONSTANT_ICONV,
  OP_CONSTANT_ONE,
  OP_CONSTANT_POOL,
  OP_SQUEEZE,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze
  OP_UNSQUEEZE,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
  OP_EW_SUB,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
  OP_EW_DIV,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
  OP_EW_EQUAL,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
  OP_EW_GREATER,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater
  OP_EW_LESS,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less
  OP_EW_MAX,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max
  OP_EW_MIN,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min
  OP_RECIPROCAL,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reciprocal
  OP_REDUCE_ARGMAX,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax
  OP_REDUCE_ARGMIN,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin
  OP_REDUCE_MAX,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax
  OP_REDUCE_MEAN,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean
  OP_REDUCE_MIN,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMin
  OP_REDUCE_PROD,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceProd
  OP_REDUCE_SUM,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum
  OP_PAD,  // https://github.com/dmlc/tvm/blob/master/topi/python/topi/nn/pad.py
  OP_SHAPE,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape
  OP_SIZE,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Size
  OP_TOPK,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
  OP_WHERE,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where
  OP_CEIL,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil
  OP_CAST,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
  OP_EXP,    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp
  OP_ROUND,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round
  OP_LOG,    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
  OP_LOGICAL_NOT,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not
  OP_SQRT,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt
  OP_LEAKYRELU,
  OP_SLICE,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice
  OP_RESIZE,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
  OP_PRELU,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu
  OP_GELU,
  OP_MULTIHEAD_ATTENTION,
  OP_INC_MULTIHEAD_SELF_ATTENTION,
  OP_FUSED,  // Fused operator type for internal fusion optimizations
  // Parallel Ops
  OP_REPARTITION,
  OP_COMBINE,
  OP_REPLICATE,
  OP_REDUCTION,
  OP_PIPELINE,
  OP_FUSED_PARALLEL,
};

enum LayerTaskID {
  BINARY_TASK_ID,
  CONCAT_TASK_ID,
  CONV2D_TASK_ID,
  MATMUL_TASK_ID,
  RESHAPE_TASK_ID,
  SOFTMAX_TASK_ID,
  UNARY_TASK_ID,
};

// forward declarations of some types
class Tensor;
class Weights;
class Operator;
struct InputTensor;
struct OutputTensor;
struct PartitionStrategy;
class LegionModelState;
class LegionModelInstance;
class LegionTritonRuntime;

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_TYPES_H__
