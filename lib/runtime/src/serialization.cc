#include "serialization.h"

using Legion::Deserializer;
using Legion::Serializer;

namespace FlexFlow {

bool needs_serialize(SearchSolution const &s) {
  return visit_needs_serialize(s);
}

void serialize(Serializer &sez, SearchSolution const &s) {
  return visit_serialize(sez, s);
}

void deserialize(Deserializer &dez, SearchSolution &s) {
  return visit_deserialize(dez, s);
}

} // namespace FlexFlow
