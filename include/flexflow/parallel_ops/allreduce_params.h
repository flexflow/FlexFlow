#ifndef _FLEXFLOW_ALLREDUCE_PARAMS_H
#define _FLEXFLOW_ALLREDUCE_PARAMS_H

namespace FlexFlow {

struct AllReduceParams {
  int allreduce_legion_dim;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(AllReduceParams const &, AllReduceParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::AllReduceParams> {
  size_t operator()(FlexFlow::AllReduceParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_ALLREDUCE_PARAMS_H
