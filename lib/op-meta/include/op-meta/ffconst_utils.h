#ifndef _FLEXFLOW_FFCONST_UTILS_H
#define _FLEXFLOW_FFCONST_UTILS_H

#include "op-meta/op-meta.h"
#include <string>

namespace FlexFlow {

std::string get_operator_type_name(OperatorType type);
bool is_parallel_op(OperatorType const &);

std::ostream &operator<<(std::ostream &, OperatorType);

}; // namespace FlexFlow

#endif // _FLEXFLOW_FFCONST_UTILS_H
