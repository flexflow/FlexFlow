#include "utils/graph/multidiedge.h"

namespace FlexFlow {

bool MultiDiOutput::operator>(MultiDiOutput const &other) const {
  return !(*this < other) && !(*this == other);
}

bool MultiDiOutput::operator>=(MultiDiOutput const &other) const {
  return !(*this < other);
}

bool MultiDiOutput::operator<=(MultiDiOutput const &other) const {
  return (*this < other) || (*this == other);
}

} // namespace FlexFlow
