#ifndef _FLEXFLOW_PARTITION_PARAMS_H
#define _FLEXFLOW_PARTITION_PARAMS_H

namespace FlexFlow {

struct RepartitionParams {
  int repartition_legion_dim;
  int repartition_degree;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(RepartitionParams const &, RepartitionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::RepartitionParams> {
  size_t operator()(FlexFlow::RepartitionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_PARTITION_PARAMS_H
