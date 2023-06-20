#ifndef _FLEXFLOW_FFCONST_UTILS_H
#define _FLEXFLOW_FFCONST_UTILS_H

#include "flexflow/ffconst.h"
#include <string>

namespace FlexFlow {

std::string get_operator_type_name(OperatorType type);

size_t data_type_size(DataType type);

#define INT4_NUM_OF_ELEMENTS_PER_GROUP 32

size_t get_quantization_to_byte_size(DataType type,
                                     DataType quantization_type,
                                     size_t num_elements);

std::ostream &operator<<(std::ostream &, OperatorType);

}; // namespace FlexFlow

#endif // _FLEXFLOW_FFCONST_UTILS_H
