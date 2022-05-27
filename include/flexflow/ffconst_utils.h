#ifndef _FLEXFLOW_FFCONST_UTILS_H
#define _FLEXFLOW_FFCONST_UTILS_H

#include "flexflow/ffconst.h"
#include <string>

namespace FlexFlow {

std::string get_operator_type_name(OperatorType type);

std::ostream &operator<<(std::ostream &, OperatorType);

}; // namespace FlexFlow

#endif // _FLEXFLOW_FFCONST_UTILS_H