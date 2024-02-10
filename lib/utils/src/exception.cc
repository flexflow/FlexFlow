#include "utils/exception.h"

namespace FlexFlow {

not_implemented::not_implemented()
    : std::logic_error("Function not yet implemented"){};

}
