#include "realm_allocator.h"

using Legion::coord_t;

namespace FlexFlow {

RealmAllocator::RealmAllocator(Legion::Memory memory) 
  : memory(memory), instances{} 
{ }

void *RealmAllocator::allocate(size_t size) {
  Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
                                 Realm::Point<1, coord_t>(size - 1));
  this->instances.emplace_back();
  Realm::RegionInstance::create_instance(this->instances.back(),
                                         this->memory,
                                         bounds,
                                         {sizeof(char)},
                                         0,
                                         Realm::ProfilingRequestSet()).wait();
  return this->instances.back().pointer_untyped(0, sizeof(char));
}

RealmAllocator::~RealmAllocator() {
  for (auto &instance : this->instances) {
    instance.destroy();
  }
}

}
