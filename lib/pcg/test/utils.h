#ifndef _FLEXFLOW_PCG_TEST_UTILS_H
#define _FLEXFLOW_PCG_TEST_UTILS_H

#include "utils/json.h"

namespace FlexFlow {

std::string str(json const &j);

using Field = std::pair<std::string, std::string>;
void check_fields(json const &j, std::vector<Field> const &fields);

} // namespace FlexFlow

#endif // _FLEXFLOW_PCG_TEST_UTILS_H
