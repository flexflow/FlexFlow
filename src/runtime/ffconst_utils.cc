#include "flexflow/ffconst_utils.h"
#include "flexflow/accessor.h"
#include <stdexcept>

namespace FlexFlow {

std::string get_operator_type_name(OperatorType type) {
  switch (type) {
    case OP_CONV2D:
      return "Conv2D";
    case OP_DROPOUT:
      return "Dropout";
    case OP_LINEAR:
      return "Dense";
    case OP_BATCHMATMUL:
      return "BatchMatMul";
    case OP_POOL2D:
      return "Pool2D";
    case OP_SCALAR_MULTIPLY:
      return "ScalarMultiply";
    case OP_SCALAR_ADD:
      return "ScalarAdd";
    case OP_SCALAR_FLOOR_DIV:
      return "ScalarFloorDiv";
    case OP_SCALAR_TRUE_DIV:
      return "ScalarTrueDiv";
    case OP_SCALAR_SUB:
      return "ScalarSub";
    case OP_RELU:
      return "ReLU";
    case OP_SIGMOID:
      return "Sigmoid";
    case OP_TANH:
      return "Tanh";
    case OP_ELU:
      return "Elu";
    case OP_FLAT:
      return "Flat";
    case OP_SOFTMAX:
      return "Softmax";
    case OP_BATCHNORM:
      return "BatchNorm";
    case OP_CONCAT:
      return "Concat";
    case OP_SPLIT:
      return "Split";
    case OP_EMBEDDING:
      return "Embedding";
    case OP_EXPERTS:
      return "Experts";
    case OP_GATHER:
      return "Gather";
    case OP_GROUP_BY:
      return "Group_by";
    case OP_CACHE:
      return "Cache";
    case OP_AGGREGATE:
      return "Aggregate cooperation";
    case OP_AGG_SPEC:
      return "Aggregate specification";
    case OP_RESHAPE:
      return "Reshape";
    case OP_REVERSE:
      return "Reverse";
    case OP_TRANSPOSE:
      return "Transpose";
    case OP_EW_ADD:
      return "Add";
    case OP_EW_MUL:
      return "Mul";
    case OP_MATMUL:
      return "Matmul";
    case OP_MUL:
      return "Mul";
    case OP_ENLARGE:
      return "Enlarge";
    case OP_SQUEEZE:
      return "Squeeze";
    case OP_UNSQUEEZE:
      return "Unsqueeze";
    case OP_EW_SUB:
      return "Sub";
    case OP_EW_DIV:
      return "Div";
    case OP_EW_EQUAL:
      return "Equal";
    case OP_EW_GREATER:
      return "Greater";
    case OP_EW_LESS:
      return "Less";
    case OP_EW_MAX:
      return "Max";
    case OP_EW_MIN:
      return "Min";
    case OP_REDUCE_ARGMAX:
      return "ReduceArgMax";
    case OP_REDUCE_ARGMIN:
      return "ReduceArgMin";
    case OP_REDUCE_MAX:
      return "ReduceMax";
    case OP_REDUCE_MEAN:
      return "ReduceMean";
    case OP_REDUCE_MIN:
      return "ReduceMin";
    case OP_REDUCE_PROD:
      return "ReduceProd";
    case OP_REDUCE_SUM:
      return "ReduceSum";
    case OP_PAD:
      return "Pad";
    case OP_SHAPE:
      return "Shape";
    case OP_SIZE:
      return "Size";
    case OP_TOPK:
      return "TopK";
    case OP_ARG_TOPK:
      return "ArgTopK";
    case OP_BEAM_TOPK:
      return "BeamTopK";
    case OP_WHERE:
      return "Where";
    case OP_CEIL:
      return "Ceil";
    case OP_CAST:
      return "Cast";
    case OP_EXP:
      return "Exp";
    case OP_SIN:
      return "Sin";
    case OP_COS:
      return "Cos";
    case OP_ROUND:
      return "Round";
    case OP_LOG:
      return "Log";
    case OP_LOGICAL_NOT:
      return "LogicalNot";
    case OP_SQRT:
      return "Sqrt";
    case OP_LEAKYRELU:
      return "LeakyReLU";
    case OP_SLICE:
      return "Slice";
    case OP_RESIZE:
      return "Resize";
    case OP_PRELU:
      return "PReLU";
    case OP_MULTIHEAD_ATTENTION:
      return "MultiHeadAttention";
    case OP_INC_MULTIHEAD_SELF_ATTENTION:
      return "IncMultiHeadSelfAttention";
    case OP_SPEC_INC_MULTIHEAD_SELF_ATTENTION:
      return "SpecIncMultiHeadSelfAttention";
    case OP_TREE_INC_MULTIHEAD_SELF_ATTENTION:
      return "TreeIncMultiHeadSelfAttention";
    case OP_INPUT:
      return "Input";
    case OP_WEIGHT:
      return "Weight";
    case OP_NOOP:
      return "NoOp";
    case OP_FUSED:
      return "FusedOp";
    case OP_RSQRT:
      return "Rsqrt";
    case OP_POW:
      return "Pow";
    case OP_MEAN:
      return "Mean";
    case OP_LAYERNORM:
      return "LayerNorm";
    case OP_RESIDUAL_LAYERNORM:
      return "ResidualLayerNorm";
    case OP_ADD_BIAS_RESIDUAL_LAYERNORM:
      return "AddBiasResidualLayerNorm";
    case OP_SIGMOID_SILU_MULTI:
      return "SigmoidSiluMulti";
    case OP_RMS_NORM:
      return "RMSNorm";
    case OP_RESIDUAL_RMS_NORM:
      return "ResidualRMSNorm";
    case OP_GELU:
      return "GELU";
    case OP_IDENTITY:
      return "Identity";
    case OP_SAMPLING:
      return "Sampling";
    case OP_ARGMAX:
      return "ArgMax";
    // PEFT Ops
    case OP_LORA:
      return "Lora Layer";
    // Parallel Ops
    case OP_REPARTITION:
      return "Repartition";
    case OP_COMBINE:
      return "Combine";
    case OP_REPLICATE:
      return "Replicate";
    case OP_REDUCTION:
      return "Reduction";
    case OP_ALLREDUCE:
      return "AllReduce";
    case OP_PARALLEL_IDENTITY:
      return "ParallelIdentity";
    case OP_PIPELINE:
      return "Pipeline";
    case OP_FUSED_PARALLEL:
      return "FusedParallelOp";
    default:
      throw std::runtime_error("Operator type unsupported: " +
                               std::to_string(type));
  }
}

size_t data_type_size(DataType type) {
  switch (type) {
    case DT_HALF:
      return sizeof(half);
    case DT_FLOAT:
      return sizeof(float);
    case DT_DOUBLE:
      return sizeof(double);
    case DT_INT32:
      return sizeof(int32_t);
    case DT_INT64:
      return sizeof(int64_t);
    case DT_BOOLEAN:
      return sizeof(bool);
    default:
      assert(false);
  }
}

size_t get_quantization_to_byte_size(DataType type,
                                     DataType quantization_type,
                                     size_t num_elements) {
  assert(quantization_type == DT_INT4 || quantization_type == DT_INT8);
  return (num_elements / (quantization_type == DT_INT4 ? 2 : 1)) +
         (num_elements / INT4_NUM_OF_ELEMENTS_PER_GROUP) * 2 *
             data_type_size(type);
}

std::ostream &operator<<(std::ostream &s, OperatorType op_type) {
  s << get_operator_type_name(op_type);

  return s;
}

}; // namespace FlexFlow
