#include "utils/test_types/capability.h"
#include "utils/ff_exceptions/ff_exceptions.h"
#include <sstream>

namespace FlexFlow::test_types {

std::string format_as(capability_t c) {
  switch (c) {
    case HASHABLE: return "HASHABLE";
    case EQ: return "EQ";
    case CMP: return "CMP";
    case DEFAULT_CONSTRUCTIBLE: return "DEFAULT_CONSTRUCTIBLE";
    case MOVE_CONSTRUCTIBLE: return "MOVE_CONSTRUCTIBLE";
    case COPY_CONSTRUCTIBLE: return "COPY_CONSTRUCTIBLE";
    case COPY_ASSIGNABLE: return "COPY_ASSIGNABLE";
    case PLUS: return "PLUS";
    case PLUSEQ: return "PLUSEQ";
    case FMT: return "FMT";
    default:
      std::ostringstream oss;
      oss << "Unknown capability {}" << static_cast<int>(c);
      throw mk_runtime_error(oss.str());
  }
}

}
