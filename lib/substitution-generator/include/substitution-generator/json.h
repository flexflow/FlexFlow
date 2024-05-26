#ifndef _FLEXFLOW_SUBSTITUTION_LOADER_H
#define _FLEXFLOW_SUBSTITUTION_LOADER_H

#include "op-attrs/operator_type.h"
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

NLOHMANN_JSON_SERIALIZE_ENUM(
    Op,
    {{OperatorType::NOOP, "OP_NOOP"},
     {OperatorType::CONV2D, "OP_CONV2D"},
     {OperatorType::DROPOUT, "OP_DROPOUT"},
     {OperatorType::LINEAR, "OP_LINEAR"},
     {OperatorType::BATCHMATMUL, "OP_BATCHMATMUL"},
     {OperatorType::POOL2D, "OP_POOL2D_MAX"},
     {OperatorType::SCALAR_MULTIPLY, "OP_SCALAR_MULTIPLY"},
     {OperatorType::SCALAR_ADD, "OP_SCALAR_ADD"},
     {OperatorType::SCALAR_FLOOR_DIV, "OP_SCALAR_FLOOR_DIV"},
     {OperatorType::SCALAR_TRUE_DIV, "OP_SCALAR_TRUE_DIV"},
     {OperatorType::SCALAR_SUB, "OP_SCALAR_SUB"},
     {OperatorType::RELU, "OP_RELU"},
     {OperatorType::IDENTITY, "OP_IDENTITY"},
     {OperatorType::SIGMOID, "OP_SIGMOID"},
     {OperatorType::TANH, "OP_TANH"},
     {OperatorType::ELU, "OP_ELU"},
     {OperatorType::FLAT, "OP_FLAT"},
     {OperatorType::SOFTMAX, "OP_SOFTMAX"},
     {OperatorType::BATCHNORM, "OP_BATCHNORM"},
     {OperatorType::CONCAT, "OP_CONCAT"},
     {OperatorType::SPLIT, "OP_SPLIT"},
     {OperatorType::EMBEDDING, "OP_EMBEDDING"},
     {OperatorType::CACHE, "OP_CACHE"},
     {OperatorType::RESHAPE, "OP_RESHAPE"},
     {OperatorType::REVERSE, "OP_REVERSE"},
     {OperatorType::TRANSPOSE, "OP_TRANSPOSE"},
     {OperatorType::EW_ADD, "OP_EW_ADD"},
     {OperatorType::EW_MUL, "OP_EW_MUL"},
     {OperatorType::MATMUL, "OP_MATMUL"},
     {OperatorType::MUL, "OP_MUL"},
     {OperatorType::ENLARGE, "OP_ENLARGE"},
     {OperatorType::SQUEEZE, "OP_SQUEEZE"},
     {OperatorType::UNSQUEEZE, "OP_UNSQUEEZE"},
     {OperatorType::EW_SUB, "OP_EW_SUB"},
     {OperatorType::EW_DIV, "OP_EW_DIV"},
     {OperatorType::EW_EQUAL, "OP_EW_EQUAL"},
     {OperatorType::EW_GREATER, "OP_EW_GREATER"},
     {OperatorType::EW_LESS, "OP_EW_LESS"},
     {OperatorType::EW_MAX, "OP_EW_MAX"},
     {OperatorType::EW_MIN, "OP_EW_MIN"},
     {OperatorType::REDUCE_ARGMAX, "OP_REDUCE_ARGMAX"},
     {OperatorType::REDUCE_ARGMIN, "OP_REDUCE_ARGMIN"},
     {OperatorType::REDUCE_MAX, "OP_REDUCE_MAX"},
     {OperatorType::REDUCE_MEAN, "OP_REDUCE_MEAN"},
     {OperatorType::REDUCE_MIN, "OP_REDUCE_MIN"},
     {OperatorType::REDUCE_PROD, "OP_REDUCE_PROD"},
     {OperatorType::REDUCE_SUM, "OP_REDUCE_SUM"},
     {OperatorType::PAD, "OP_PAD"},
     {OperatorType::SHAPE, "OP_SHAPE"},
     {OperatorType::SIZE, "OP_SIZE"},
     {OperatorType::TOPK, "OP_TOPK"},
     {OperatorType::WHERE, "OP_WHERE"},
     {OperatorType::CEIL, "OP_CEIL"},
     {OperatorType::CAST, "OP_CAST"},
     {OperatorType::EXP, "OP_EXP"},
     {OperatorType::ROUND, "OP_ROUND"},
     {OperatorType::LOG, "OP_LOG"},
     {OperatorType::LOGICAL_NOT, "OP_LOGICAL_NOT"},
     {OperatorType::SQRT, "OP_SQRT"},
     {OperatorType::SIN, "OP_SIN"},
     {OperatorType::COS, "OP_COS"},
     {OperatorType::LEAKYRELU, "OP_LEAKYRELU"},
     {OperatorType::SLICE, "OP_SLICE"},
     {OperatorType::RESIZE, "OP_RESIZE"},
     {OperatorType::PRELU, "OP_PRELU"},
     {OperatorType::GELU, "OP_GELU"},
     {OperatorType::MULTIHEAD_ATTENTION, "OP_MULTIHEAD_ATTENTION"},
     {OperatorType::FUSED, "OP_FUSED"},
     {OperatorType::RSQRT, "OP_RSQRT"},
     {OperatorType::POW, "OP_POW"},
     {OperatorType::MEAN, "OP_MEAN"},
     {OperatorType::LAYERNORM, "OP_LAYERNORM"},
     {OperatorType::REPARTITION, "OP_PARTITION"},
     {OperatorType::COMBINE, "OP_COMBINE"},
     {OperatorType::REPLICATE, "OP_REPLICATE"},
     {OperatorType::REDUCTION, "OP_REDUCE"},
     {OperatorType::PIPELINE, "OP_PIPELINE"},
     {OperatorType::FUSED_PARALLEL, "OP_FUSED_PARALLEL"}})

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
