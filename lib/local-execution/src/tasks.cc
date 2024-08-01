#include "local-execution/tasks.h"
#include "utils/fmt.h"

namespace FlexFlow {

std::string format_as(task_id_t const &x) {
  return fmt::format("task_id_t");
}

std::ostream &operator<<(std::ostream &s, task_id_t const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow
