#include "utils/fmt/json.h"

namespace fmt {

template
  struct formatter<::nlohmann::json, char>;

}
