#ifndef _FLEXFLOW_SUBSTITUTION_LOADER_H
#define _FLEXFLOW_SUBSTITUTION_LOADER_H

#include "op-attrs/op.h"
#include <fstream>
#include <nlohmann/json.hpp>

namespace FlexFlow {

enum PMParameter {
  PM_OP_TYPE,            // AnyOp
  PM_NUM_INPUTS,         // AnyOp
  PM_NUM_OUTPUTS,        // AnyOp
  PM_GROUP,              // Conv2D
  PM_KERNEL_H,           // Conv2D, Pool2D
  PM_KERNEL_W,           // Conv2D, Pool2D
  PM_STRIDE_H,           // Conv2D, Pool2D
  PM_STRIDE_W,           // Conv2D, Pool2D
  PM_PADDING_H,          // Conv2D, Pool2D
  PM_PADDING_W,          // Conv2D, Pool2D
  PM_ACTI,               // Conv2D, Pool2D
  PM_NUMDIM,             // Concat, Transpose
  PM_AXIS,               // Concat, Split
  PM_PERM,               // Transpose
  PM_OUTSHUFFLE,         // Transpose
  PM_MERGE_GCONV_COUNT,  // MergeGConv
  PM_AXES,               // Squeeze, Unsqueeze, Reduce*
  PM_KEEP_DIMS,          // Reduce*
  PM_EPSILON,            // BatchNorm
  PM_REPARTITION_DIM,    // Repartition
  PM_REPARTITION_DEGREE, // Repartition
  PM_REPLICATE_DIM,      // Replicate
  PM_REPLICATE_DEGREE,   // Replicate
  PM_COMBINE_DIM,        // Combine
  PM_COMBINE_DEGREE,     // Combine
  PM_REDUCTION_DIM,      // Reduction
  PM_REDUCTION_DEGREE,   // Reduction
  PM_SOFTMAX_DIM,        // Softmax
  PM_NUM_HEADS,          // MultiHeadAttention
  PM_INVALID,
  PM_PARALLEL_DIM,
  PM_PARALLEL_DEGREE,
  PM_PAD,
};

NLOHMANN_JSON_SERIALIZE_ENUM(PMParameter,
                             {{PM_INVALID, nullptr},
                              {PM_OP_TYPE, "PM_OP_TYPE"},
                              {PM_NUM_INPUTS, "PM_NUM_INPUTS"},
                              {PM_NUM_OUTPUTS, "PM_NUM_OUTPUTS"},
                              {PM_GROUP, "PM_GROUP"},
                              {PM_KERNEL_H, "PM_KERNEL_H"},
                              {PM_KERNEL_W, "PM_KERNEL_W"},
                              {PM_STRIDE_H, "PM_STRIDE_H"},
                              {PM_STRIDE_W, "PM_STRIDE_W"},
                              {PM_PADDING_H, "PM_PADDING_H"},
                              {PM_PADDING_W, "PM_PADDING_W"},
                              {PM_ACTI, "PM_ACTI"},
                              {PM_NUMDIM, "PM_NUMDIM"},
                              {PM_AXIS, "PM_AXIS"},
                              {PM_PERM, "PM_PERM"},
                              {PM_OUTSHUFFLE, "PM_OUTSHUFFLE"},
                              {PM_MERGE_GCONV_COUNT, "PM_MERGE_GCONV_COUNT"},
                              {PM_AXES, "PM_AXES"},
                              {PM_KEEP_DIMS, "PM_KEEP_DIMS"},
                              {PM_EPSILON, "PM_EPSILON"},
                              {PM_REPARTITION_DIM, "PM_REPARTITION_DIM"},
                              {PM_REPARTITION_DEGREE, "PM_REPARTITION_DEGREE"},
                              {PM_REPLICATE_DIM, "PM_REPLICATE_DIM"},
                              {PM_REPLICATE_DEGREE, "PM_REPLICATE_DEGREE"},
                              {PM_COMBINE_DIM, "PM_COMBINE_DIM"},
                              {PM_COMBINE_DEGREE, "PM_COMBINE_DEGREE"},
                              {PM_REDUCTION_DIM, "PM_REDUCTION_DIM"},
                              {PM_REDUCTION_DEGREE, "PM_REDUCTION_DEGREE"},
                              {PM_SOFTMAX_DIM, "PM_SOFTMAX_DIM"},
                              {PM_NUM_HEADS, "PM_NUM_HEADS"},
                              {PM_PARALLEL_DIM, "PM_PARALLEL_DIM"},
                              {PM_PARALLEL_DEGREE, "PM_PARALLEL_DEGREE"},
                              {PM_PAD, "PM_PAD"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Op,
                             {{Op::NOOP, "OP_NOOP"},
                              {Op::CONV2D, "OP_CONV2D"},
                              {Op::DROPOUT, "OP_DROPOUT"},
                              {Op::LINEAR, "OP_LINEAR"},
                              {Op::BATCHMATMUL, "OP_BATCHMATMUL"},
                              {Op::POOL2D, "OP_POOL2D_MAX"},
                              {Op::SCALAR_MULTIPLY, "OP_SCALAR_MULTIPLY"},
                              {Op::SCALAR_ADD, "OP_SCALAR_ADD"},
                              {Op::SCALAR_FLOOR_DIV, "OP_SCALAR_FLOOR_DIV"},
                              {Op::SCALAR_TRUE_DIV, "OP_SCALAR_TRUE_DIV"},
                              {Op::SCALAR_SUB, "OP_SCALAR_SUB"},
                              {Op::RELU, "OP_RELU"},
                              {Op::IDENTITY, "OP_IDENTITY"},
                              {Op::SIGMOID, "OP_SIGMOID"},
                              {Op::TANH, "OP_TANH"},
                              {Op::ELU, "OP_ELU"},
                              {Op::FLAT, "OP_FLAT"},
                              {Op::SOFTMAX, "OP_SOFTMAX"},
                              {Op::BATCHNORM, "OP_BATCHNORM"},
                              {Op::CONCAT, "OP_CONCAT"},
                              {Op::SPLIT, "OP_SPLIT"},
                              {Op::EMBEDDING, "OP_EMBEDDING"},
                              {Op::CACHE, "OP_CACHE"},
                              {Op::RESHAPE, "OP_RESHAPE"},
                              {Op::REVERSE, "OP_REVERSE"},
                              {Op::TRANSPOSE, "OP_TRANSPOSE"},
                              {Op::EW_ADD, "OP_EW_ADD"},
                              {Op::EW_MUL, "OP_EW_MUL"},
                              {Op::MATMUL, "OP_MATMUL"},
                              {Op::MUL, "OP_MUL"},
                              {Op::ENLARGE, "OP_ENLARGE"},
                              {Op::SQUEEZE, "OP_SQUEEZE"},
                              {Op::UNSQUEEZE, "OP_UNSQUEEZE"},
                              {Op::EW_SUB, "OP_EW_SUB"},
                              {Op::EW_DIV, "OP_EW_DIV"},
                              {Op::EW_EQUAL, "OP_EW_EQUAL"},
                              {Op::EW_GREATER, "OP_EW_GREATER"},
                              {Op::EW_LESS, "OP_EW_LESS"},
                              {Op::EW_MAX, "OP_EW_MAX"},
                              {Op::EW_MIN, "OP_EW_MIN"},
                              {Op::REDUCE_ARGMAX, "OP_REDUCE_ARGMAX"},
                              {Op::REDUCE_ARGMIN, "OP_REDUCE_ARGMIN"},
                              {Op::REDUCE_MAX, "OP_REDUCE_MAX"},
                              {Op::REDUCE_MEAN, "OP_REDUCE_MEAN"},
                              {Op::REDUCE_MIN, "OP_REDUCE_MIN"},
                              {Op::REDUCE_PROD, "OP_REDUCE_PROD"},
                              {Op::REDUCE_SUM, "OP_REDUCE_SUM"},
                              {Op::PAD, "OP_PAD"},
                              {Op::SHAPE, "OP_SHAPE"},
                              {Op::SIZE, "OP_SIZE"},
                              {Op::TOPK, "OP_TOPK"},
                              {Op::WHERE, "OP_WHERE"},
                              {Op::CEIL, "OP_CEIL"},
                              {Op::CAST, "OP_CAST"},
                              {Op::EXP, "OP_EXP"},
                              {Op::ROUND, "OP_ROUND"},
                              {Op::LOG, "OP_LOG"},
                              {Op::LOGICAL_NOT, "OP_LOGICAL_NOT"},
                              {Op::SQRT, "OP_SQRT"},
                              {Op::SIN, "OP_SIN"},
                              {Op::COS, "OP_COS"},
                              {Op::LEAKYRELU, "OP_LEAKYRELU"},
                              {Op::SLICE, "OP_SLICE"},
                              {Op::RESIZE, "OP_RESIZE"},
                              {Op::PRELU, "OP_PRELU"},
                              {Op::GELU, "OP_GELU"},
                              {Op::MULTIHEAD_ATTENTION,
                               "OP_MULTIHEAD_ATTENTION"},
                              {Op::FUSED, "OP_FUSED"},
                              {Op::RSQRT, "OP_RSQRT"},
                              {Op::POW, "OP_POW"},
                              {Op::MEAN, "OP_MEAN"},
                              {Op::LAYERNORM, "OP_LAYERNORM"},
                              {Op::REPARTITION, "OP_PARTITION"},
                              {Op::COMBINE, "OP_COMBINE"},
                              {Op::REPLICATE, "OP_REPLICATE"},
                              {Op::REDUCTION, "OP_REDUCE"},
                              {Op::PIPELINE, "OP_PIPELINE"},
                              {Op::FUSED_PARALLEL, "OP_FUSED_PARALLEL"}})

struct Parameter {
  PMParameter key;
  int value;
};
void from_json(nlohmann::json const &j, Parameter &p);

struct Tensor {
  int opId;
  int tsId;
};
void from_json(nlohmann::json const &j, Tensor &t);

struct Operator {
  OperatorType op_type;
  std::vector<Tensor> input;
  std::vector<Parameter> para;

  std::optional<int> at(PMParameter key) const;
};
void from_json(nlohmann::json const &j, Operator &t);

struct MapOutput {
  int dstOpId;
  int dstTsId;
  int srcOpId;
  int srcTsId;
};
void from_json(nlohmann::json const &j, MapOutput &t);

struct Rule {
  std::string name;
  std::vector<Operator> srcOp;
  std::vector<Operator> dstOp;
  std::vector<MapOutput> mappedOutput;
};
void from_json(nlohmann::json const &j, Rule &t);

struct RuleCollection {
  std::vector<Rule> rules;
};
void from_json(nlohmann::json const &j, RuleCollection &c);

RuleCollection load_rule_collection(std::istream &s);
RuleCollection load_rule_collection_from_path(std::string const &path);

} // namespace FlexFlow

#endif // _FLEXFLOW_SUBSTITUTION_LOADER_H
