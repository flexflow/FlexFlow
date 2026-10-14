#include "realm-execution/tasks/serializer/serializable_realm_instance.h"
#include <libassert/assert.hpp>
#include <type_traits>

namespace FlexFlow {

// Realm::RegionInstance is trivially copyable so it's safe to treat it as bytes
static_assert(std::is_trivially_copy_constructible_v<Realm::RegionInstance>);

SerializableRealmInstance
    realm_instance_to_serializable(Realm::RegionInstance const &inst) {
  uint8_t const *data = reinterpret_cast<uint8_t const *>(&inst);
  return SerializableRealmInstance{
      std::vector<uint8_t>{data, data + sizeof(inst)}};
}

Realm::RegionInstance
    realm_instance_from_serializable(SerializableRealmInstance const &inst) {
  ASSERT(inst.instance.size() == sizeof(Realm::RegionInstance));
  return *reinterpret_cast<Realm::RegionInstance const *>(inst.instance.data());
}

} // namespace FlexFlow
