#include "local-execution/task_argument_accessor.h"

namespace FlexFlow {
  
struct FwdBwdTaskImplFunction {

std::optional<float> (*fwd_bwd_task_impl_function)(TaskArgumentAccessor const &);

bool operator==(FwdBwdTaskImplFunction const &) const;
bool operator!=(FwdBwdTaskImplFunction const &) const;

};

std::string format_as(FwdBwdTaskImplFunction const &x);
std::ostream &operator<<(std::ostream &s, FwdBwdTaskImplFunction const &x);

} // namespace FlexFlow

