#ifndef _FLEXFLOW_REPLICATE_PARAMS_H
#define _FLEXFLOW_REPLICATE_PARAMS_H

namespace FlexFlow {

struct ReplicateParams {
  int replicate_legion_dim;
  int replicate_degree;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(ReplicateParams const &, ReplicateParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ReplicateParams> {
  size_t operator()(FlexFlow::ReplicateParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_REPLICATE_PARAMS_H
