#include "serialization.h"

using Legion::Serializer;
using Legion::Deserializer;

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


}
