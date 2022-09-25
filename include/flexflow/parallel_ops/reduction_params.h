#ifndef _FLEXFLOW_REDUCTION_PARAMS_H
#define _FLEXFLOW_REDUCTION_PARAMS_H

namespace FlexFlow {

struct ReductionParams {
  int reduction_legion_dim;
  int reduction_degree;
};
bool operator==(ReductionParams const &, ReductionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ReductionParams> {
  size_t operator()(FlexFlow::ReductionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_REDUCTION_PARAMS_H
