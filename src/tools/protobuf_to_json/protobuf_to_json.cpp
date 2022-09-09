#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>

#include "rules.pb.h"
#include <nlohmann/json.hpp>

namespace gs = GraphSubst;
using json = nlohmann::json;

enum class OpType {
  OP_INPUT = 0,
  OP_WEIGHT = 1,
  OP_ANY = 2,
  OP_CONV2D = 3,
  OP_DROPOUT = 4,
  OP_LINEAR = 5,
  OP_POOL2D_MAX = 6,
  OP_POOL2D_AVG = 7,
  OP_RELU = 8,
  OP_SIGMOID = 9,
  OP_TANH = 10,
  OP_BATCHNORM = 11,
  OP_CONCAT = 12,
  OP_SPLIT = 13,
  OP_RESHAPE = 14,
  OP_TRANSPOSE = 15,
  OP_EW_ADD = 16,
  OP_EW_MUL = 17,
  OP_MATMUL = 18,
  OP_MUL = 19,
  OP_ENLARGE = 20,
  OP_MERGE_GCONV = 21,
  OP_CONSTANT_IMM = 22,
  OP_CONSTANT_ICONV = 23,
  OP_CONSTANT_ONE = 24,
  OP_CONSTANT_POOL = 25,
  OP_PARTITION = 26,
  OP_COMBINE = 27,
  OP_REPLICATE = 28,
  OP_REDUCE = 29,
  OP_EMBEDDING = 30
};

NLOHMANN_JSON_SERIALIZE_ENUM(OpType,
                             {{OpType::OP_INPUT, "OP_INPUT"},
                              {OpType::OP_WEIGHT, "OP_WEIGHT"},
                              {OpType::OP_ANY, "OP_ANY"},
                              {OpType::OP_CONV2D, "OP_CONV2D"},
                              {OpType::OP_DROPOUT, "OP_DROPOUT"},
                              {OpType::OP_LINEAR, "OP_LINEAR"},
                              {OpType::OP_POOL2D_MAX, "OP_POOL2D_MAX"},
                              {OpType::OP_POOL2D_AVG, "OP_POOL2D_AVG"},
                              {OpType::OP_RELU, "OP_RELU"},
                              {OpType::OP_SIGMOID, "OP_SIGMOID"},
                              {OpType::OP_TANH, "OP_TANH"},
                              {OpType::OP_BATCHNORM, "OP_BATCHNORM"},
                              {OpType::OP_CONCAT, "OP_CONCAT"},
                              {OpType::OP_SPLIT, "OP_SPLIT"},
                              {OpType::OP_RESHAPE, "OP_RESHAPE"},
                              {OpType::OP_TRANSPOSE, "OP_TRANSPOSE"},
                              {OpType::OP_EW_ADD, "OP_EW_ADD"},
                              {OpType::OP_EW_MUL, "OP_EW_MUL"},
                              {OpType::OP_MATMUL, "OP_MATMUL"},
                              {OpType::OP_MUL, "OP_MUL"},
                              {OpType::OP_ENLARGE, "OP_ENLARGE"},
                              {OpType::OP_MERGE_GCONV, "OP_MERGE_GCONV"},
                              {OpType::OP_CONSTANT_IMM, "OP_CONSTANT_IMM"},
                              {OpType::OP_CONSTANT_ICONV, "OP_CONSTANT_ICONV"},
                              {OpType::OP_CONSTANT_ONE, "OP_CONSTANT_ONE"},
                              {OpType::OP_CONSTANT_POOL, "OP_CONSTANT_POOl"},
                              {OpType::OP_PARTITION, "OP_PARTITION"},
                              {OpType::OP_COMBINE, "OP_COMBINE"},
                              {OpType::OP_REPLICATE, "OP_REPLICATE"},
                              {OpType::OP_REDUCE, "OP_REDUCE"},
                              {OpType::OP_EMBEDDING, "OP_EMBEDDING"}})

enum class ParamType {
  PM_OP_TYPE = 0,
  PM_NUM_INPUTS = 1,
  PM_NUM_OUTPUTS = 2,
  PM_GROUP = 3,
  PM_KERNEL_H = 4,
  PM_KERNEL_W = 5,
  PM_STRIDE_H = 6,
  PM_STRIDE_W = 7,
  PM_PAD = 8,
  PM_ACTI = 9,
  PM_NUMDIM = 10,
  PM_AXIS = 11,
  PM_PERM = 12,
  PM_OUTSHUFFLE = 13,
  PM_MERGE_GCONV_COUNT = 14,
  PM_PARALLEL_DIM = 15,
  PM_PARALLEL_DEGREE = 16,
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    ParamType,
    {{ParamType::PM_OP_TYPE, "PM_OP_TYPE"},
     {ParamType::PM_NUM_INPUTS, "PM_NUM_INPUTS"},
     {ParamType::PM_NUM_OUTPUTS, "PM_NUM_OUTPUTS"},
     {ParamType::PM_GROUP, "PM_GROUP"},
     {ParamType::PM_KERNEL_H, "PM_KERNEL_H"},
     {ParamType::PM_KERNEL_W, "PM_KERNEL_W"},
     {ParamType::PM_STRIDE_H, "PM_STRIDE_H"},
     {ParamType::PM_STRIDE_W, "PM_STRIDE_W"},
     {ParamType::PM_PAD, "PM_PAD"},
     {ParamType::PM_ACTI, "PM_ACTI"},
     {ParamType::PM_NUMDIM, "PM_NUMDIM"},
     {ParamType::PM_AXIS, "PM_AXIS"},
     {ParamType::PM_PERM, "PM_PERM"},
     {ParamType::PM_OUTSHUFFLE, "PM_OUTSHUFFLE"},
     {ParamType::PM_MERGE_GCONV_COUNT, "PM_MERGE_GCONV_COUNT"},
     {ParamType::PM_PARALLEL_DIM, "PM_PARALLEL_DIM"},
     {ParamType::PM_PARALLEL_DEGREE, "PM_PARALLEL_DEGREE"}})

enum class ActivationMode {
  AC_MODE_NONE = 0,
  AC_MODE_SIGMOID = 1,
  AC_MODE_RELU = 2,
  AC_MODE_TANH = 3
};

enum class PaddingMode { PD_MODE_SAME = 0, PD_MODE_VALID = 1 };

// partial specialization (full specialization works too)
namespace nlohmann {
template <>
struct adl_serializer<gs::Tensor> {
  static void to_json(json &j, gs::Tensor const &t) {
    j = json{{"_t", "Tensor"}, {"opId", t.opid()}, {"tsId", t.tsid()}};
  }
};

template <>
struct adl_serializer<gs::Parameter> {
  static void to_json(json &j, gs::Parameter const &p) {
    j = json{
        {"_t", "Parameter"},
        {"key", static_cast<ParamType>(p.key())},
    };
    ParamType key = static_cast<ParamType>(p.key());
    switch (key) {
      case ParamType::PM_ACTI:
        j["value"] = static_cast<ActivationMode>(p.value());
        break;
      case ParamType::PM_PAD:
        j["value"] = static_cast<PaddingMode>(p.value());
        break;
      default:
        j["value"] = p.value();
    }
  }
};

template <typename T>
struct adl_serializer<::google::protobuf::RepeatedPtrField<T>> {
  static void to_json(json &j,
                      ::google::protobuf::RepeatedPtrField<T> const &fs) {
    j = std::vector<json>{};
    for (auto const &i : fs) {
      json j2 = i;
      j.push_back(j2);
    }
  }
};

template <>
struct adl_serializer<gs::Operator> {
  static void to_json(json &j, gs::Operator const &o) {
    j = json{
        {"_t", "Operator"},
        {"type", static_cast<OpType>(o.type())},
        {"input", o.input()},
        {"para", o.para()},
    };
  }
};

template <>
struct adl_serializer<gs::MapOutput> {
  static void to_json(json &j, gs::MapOutput const &m) {
    j = json{{"_t", "MapOutput"},
             {"srcOpId", m.srcopid()},
             {"dstOpId", m.dstopid()},
             {"srcTsId", m.srctsid()},
             {"dstTsId", m.dsttsid()}};
  }
};

template <>
struct adl_serializer<gs::Rule> {
  static void to_json(json &j, gs::Rule const &r) {
    j = json{{"_t", "Rule"},
             {"srcOp", r.srcop()},
             {"dstOp", r.dstop()},
             {"mappedOutput", r.mappedoutput()}};
  }
};

template <>
struct adl_serializer<gs::RuleCollection> {
  static void to_json(json &j, gs::RuleCollection const &c) {
    j = json{{"_t", "RuleCollection"}, {"rule", c.rule()}};
    for (int i = 0; i < j["rule"].size(); ++i) {
      std::ostringstream oss;
      oss << "taso_rule_" << i;
      j["rule"][i]["name"] = oss.str();
    }
  }
};
} // namespace nlohmann

int main(int argc, char **argv) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input-file> <output-file>"
              << std::endl;
    return 1;
  }

  gs::RuleCollection rule_collection;

  std::ifstream input(argv[1], std::ios::binary);
  if (!rule_collection.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse rule collection." << std::endl;
    return -1;
  }

  std::cout << "Loaded " << rule_collection.rule_size() << " rules."
            << std::endl;

  json j = rule_collection;
  std::ofstream output(argv[2]);
  output << std::setw(2) << j << std::endl;
  return 0;
}
