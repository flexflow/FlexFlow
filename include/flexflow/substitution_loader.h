#ifndef _FLEXFLOW_SUBSTITUTION_LOADER_H
#define _FLEXFLOW_SUBSTITUTION_LOADER_H

#include "flexflow/ffconst.h"
#include "tl/optional.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

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
    OperatorType,
    {{OP_INVALID, nullptr},
     {OP_NOOP, "OP_NOOP"},
     {OP_CONV2D, "OP_CONV2D"},
     {OP_DROPOUT, "OP_DROPOUT"},
     {OP_LINEAR, "OP_LINEAR"},
     {OP_BATCHMATMUL, "OP_BATCHMATMUL"},
     {OP_POOL2D, "OP_POOL2D_MAX"},
     {OP_SCALAR_MULTIPLY, "OP_SCALAR_MULTIPLY"},
     {OP_SCALAR_ADD, "OP_SCALAR_ADD"},
     {OP_SCALAR_FLOOR_DIV, "OP_SCALAR_FLOOR_DIV"},
     {OP_SCALAR_TRUE_DIV, "OP_SCALAR_TRUE_DIV"},
     {OP_SCALAR_SUB, "OP_SCALAR_SUB"},
     {OP_RELU, "OP_RELU"},
     {OP_IDENTITY, "OP_IDENTITY"},
     {OP_SIGMOID, "OP_SIGMOID"},
     {OP_TANH, "OP_TANH"},
     {OP_ELU, "OP_ELU"},
     {OP_FLAT, "OP_FLAT"},
     {OP_SOFTMAX, "OP_SOFTMAX"},
     {OP_BATCHNORM, "OP_BATCHNORM"},
     {OP_CONCAT, "OP_CONCAT"},
     {OP_SPLIT, "OP_SPLIT"},
     {OP_EMBEDDING, "OP_EMBEDDING"},
     {OP_GROUP_BY, "OP_GROUP_BY"},
     {OP_CACHE, "OP_CACHE"},
     {OP_AGGREGATE, "OP_AGGREGATE"},
     {OP_AGG_SPEC, "OP_AGG_SPEC"},
     {OP_RESHAPE, "OP_RESHAPE"},
     {OP_REVERSE, "OP_REVERSE"},
     {OP_TRANSPOSE, "OP_TRANSPOSE"},
     {OP_EW_ADD, "OP_EW_ADD"},
     {OP_EW_MUL, "OP_EW_MUL"},
     {OP_MATMUL, "OP_MATMUL"},
     {OP_MUL, "OP_MUL"},
     {OP_ENLARGE, "OP_ENLARGE"},
     {OP_MERGE_GCONV, "OP_MERGE_GCONV"},
     {OP_CONSTANT_IMM, "OP_CONSTANT_IMM"},
     {OP_CONSTANT_ICONV, "OP_CONSTANT_ICONV"},
     {OP_CONSTANT_ONE, "OP_CONSTANT_ONE"},
     {OP_CONSTANT_POOL, "OP_CONSTANT_POOL"},
     {OP_SQUEEZE, "OP_SQUEEZE"},
     {OP_UNSQUEEZE, "OP_UNSQUEEZE"},
     {OP_EW_SUB, "OP_EW_SUB"},
     {OP_EW_DIV, "OP_EW_DIV"},
     {OP_EW_EQUAL, "OP_EW_EQUAL"},
     {OP_EW_GREATER, "OP_EW_GREATER"},
     {OP_EW_LESS, "OP_EW_LESS"},
     {OP_EW_MAX, "OP_EW_MAX"},
     {OP_EW_MIN, "OP_EW_MIN"},
     {OP_REDUCE_ARGMAX, "OP_REDUCE_ARGMAX"},
     {OP_REDUCE_ARGMIN, "OP_REDUCE_ARGMIN"},
     {OP_REDUCE_MAX, "OP_REDUCE_MAX"},
     {OP_REDUCE_MEAN, "OP_REDUCE_MEAN"},
     {OP_REDUCE_MIN, "OP_REDUCE_MIN"},
     {OP_REDUCE_PROD, "OP_REDUCE_PROD"},
     {OP_REDUCE_SUM, "OP_REDUCE_SUM"},
     {OP_PAD, "OP_PAD"},
     {OP_SHAPE, "OP_SHAPE"},
     {OP_SIZE, "OP_SIZE"},
     {OP_TOPK, "OP_TOPK"},
     {OP_WHERE, "OP_WHERE"},
     {OP_CEIL, "OP_CEIL"},
     {OP_CAST, "OP_CAST"},
     {OP_EXP, "OP_EXP"},
     {OP_ROUND, "OP_ROUND"},
     {OP_LOG, "OP_LOG"},
     {OP_LOGICAL_NOT, "OP_LOGICAL_NOT"},
     {OP_SQRT, "OP_SQRT"},
     {OP_SIN, "OP_SIN"},
     {OP_COS, "OP_COS"},
     {OP_LEAKYRELU, "OP_LEAKYRELU"},
     {OP_SLICE, "OP_SLICE"},
     {OP_RESIZE, "OP_RESIZE"},
     {OP_PRELU, "OP_PRELU"},
     {OP_GELU, "OP_GELU"},
     {OP_MULTIHEAD_ATTENTION, "OP_MULTIHEAD_ATTENTION"},
     {OP_INC_MULTIHEAD_SELF_ATTENTION, "OP_INC_MULTIHEAD_SELF_ATTENTION"},
     {OP_FUSED, "OP_FUSED"},
     {OP_RSQRT, "OP_RSQRT"},
     {OP_POW, "OP_POW"},
     {OP_MEAN, "OP_MEAN"},
     {OP_LAYERNORM, "OP_LAYERNORM"},
     {OP_RESIDUAL_LAYERNORM, "OP_RESIDUAL_LAYERNORM"},
     {OP_ADD_BIAS_RESIDUAL_LAYERNORM, "OP_ADD_BIAS_RESIDUAL_LAYERNORM"},
     {OP_SIGMOID_SILU_MULTI, "OP_SIGMOID_SILU_MULTI"},
     {OP_RMS_NORM, "OP_RMS_NORM"},
     {OP_RESIDUAL_RMS_NORM, "OP_RESIDUAL_RMS_NORM"},
     {OP_REPARTITION, "OP_PARTITION"},
     {OP_COMBINE, "OP_COMBINE"},
     {OP_REPLICATE, "OP_REPLICATE"},
     {OP_REDUCTION, "OP_REDUCE"},
     {OP_PIPELINE, "OP_PIPELINE"},
     {OP_FUSED_PARALLEL, "OP_FUSED_PARALLEL"}})

namespace FlexFlow {
namespace substitution_loader {

using json = nlohmann::json;

struct Parameter {
  PMParameter key;
  int value;
};
void from_json(json const &j, Parameter &p);

struct Tensor {
  int opId;
  int tsId;
};
void from_json(json const &j, Tensor &t);

struct Operator {
  OperatorType op_type;
  std::vector<Tensor> input;
  std::vector<Parameter> para;

  tl::optional<int> at(PMParameter key) const;
};
void from_json(json const &j, Operator &t);

struct MapOutput {
  int dstOpId;
  int dstTsId;
  int srcOpId;
  int srcTsId;
};
void from_json(json const &j, MapOutput &t);

struct Rule {
  std::string name;
  std::vector<Operator> srcOp;
  std::vector<Operator> dstOp;
  std::vector<MapOutput> mappedOutput;
};
void from_json(json const &j, Rule &t);

struct RuleCollection {
  std::vector<Rule> rules;
};
void from_json(json const &j, RuleCollection &c);

RuleCollection load_rule_collection(std::istream &s);
RuleCollection load_rule_collection_from_path(std::string const &path);

} // namespace substitution_loader
} // namespace FlexFlow

#endif // _FLEXFLOW_SUBSTITUTION_LOADER_H
