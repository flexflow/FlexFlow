#ifndef _FLEXFLOW_PARALLEL_IDENTITY_PARAMS_H
#define _FLEXFLOW_PARALLEL_IDENTITY_PARAMS_H

namespace FlexFlow {

struct ParallelIdentityParams {
  int parallel_identity_legion_dim;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(ParallelIdentityParams const &, ParallelIdentityParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ParallelIdentityParams> {
  size_t operator()(FlexFlow::ParallelIdentityParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_PARALLEL_IDENTITY_PARAMS_H
