// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/parallel_tensor_dims.struct.toml
/* proj-data
{
  "generated_from": "aec3b6b66e34be0d5ce3055822479430"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIMS_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIMS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/dim_ordered.h"
#include "op-attrs/replica_parallel_dim_set.dtg.h"
#include "op-attrs/shard_parallel_dim.dtg.h"
#include "rapidcheck.h"
#include "utils/fmt/pair.h"
#include "utils/fmt/unordered_map.h"
#include <functional>
#include <ostream>
#include <tuple>
#include <unordered_map>

namespace FlexFlow {
struct ParallelTensorDims {
  ParallelTensorDims() = delete;
  explicit ParallelTensorDims(
      ::FlexFlow::FFOrdered<::FlexFlow::ShardParallelDim> const &shard_dims,
      ::FlexFlow::ReplicaParallelDimSet const &replica_dims);

  bool operator==(ParallelTensorDims const &) const;
  bool operator!=(ParallelTensorDims const &) const;
  bool operator<(ParallelTensorDims const &) const;
  bool operator>(ParallelTensorDims const &) const;
  bool operator<=(ParallelTensorDims const &) const;
  bool operator>=(ParallelTensorDims const &) const;
  ::FlexFlow::FFOrdered<::FlexFlow::ShardParallelDim> shard_dims;
  ::FlexFlow::ReplicaParallelDimSet replica_dims;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::ParallelTensorDims> {
  size_t operator()(::FlexFlow::ParallelTensorDims const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::ParallelTensorDims> {
  static ::FlexFlow::ParallelTensorDims from_json(json const &);
  static void to_json(json &, ::FlexFlow::ParallelTensorDims const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::ParallelTensorDims> {
  static Gen<::FlexFlow::ParallelTensorDims> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(ParallelTensorDims const &);
std::ostream &operator<<(std::ostream &, ParallelTensorDims const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIMS_DTG_H
