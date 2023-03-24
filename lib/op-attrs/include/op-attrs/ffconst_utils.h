#ifndef _FLEXFLOW_FFCONST_UTILS_H
#define _FLEXFLOW_FFCONST_UTILS_H

#include "op-attrs/op-attrs.h"
#include <string>

namespace FlexFlow {

std::string get_operator_type_name(OperatorType type);
bool is_parallel_op(OperatorType const &);

std::ostream &operator<<(std::ostream &, OperatorType);

std::string get_data_type_name(DataType);
std::string to_string(DataType);

}

#endif
