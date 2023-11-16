#include "utils.h"
#include "doctest.h"
#include "utils/json.h"

namespace FlexFlow {

std::string str(json const &j) {
  std::stringstream ss;
  ss << j;
  return ss.str();
}

void check_fields(json const &j, std::vector<Field> const &fields) {
  std::string strj = str(j);
  if (fields.size()) {
    for (auto const &[key, val] : fields) {
      std::stringstream fs;
      fs << "\"" << key << "\":" << val;
      std::string field = fs.str();

      CHECK(strj.find(field) != std::string::npos);
    }
  } else {
    CHECK(strj == "null");
  }
}

} // namespace FlexFlow
