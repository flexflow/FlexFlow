#ifndef _FLEXFLOW_COMBINE_PARAMS_H
#define _FLEXFLOW_COMBINE_PARAMS_H

namespace FlexFlow {

struct CombineParams {
  int combine_legion_dim;
  int combine_degree;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(CombineParams const &, CombineParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::CombineParams> {
  size_t operator()(FlexFlow::CombineParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_COMBINE_PARAMS_H
