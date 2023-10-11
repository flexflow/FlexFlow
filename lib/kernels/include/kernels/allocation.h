#ifndef _FLEXFLOW_KERNELS_ALLOCATION_H
#define _FLEXFLOW_KERNELS_ALLOCATION_H

#include <cstddef>
#include <memory>
#include <unordered_map>

namespace FlexFlow {

struct IAllocator {
  virtual void *allocate(size_t) = 0;
  virtual void deallocate(void *) = 0;

  virtual ~IAllocator() = default;
};

struct Allocator {
  Allocator() = delete;

  void *allocate(size_t);
  void deallocate(void *);

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IAllocator, T>::value,
                                 Allocator>::type
      create(Args &&...args) {
    return Allocator(std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  std::shared_ptr<IAllocator> i_allocator;
};

struct TrackedAllocator: public Allocator {
  TrackedAllocator(Allocator const & allocator) : allocator(allocator) {};

  void *allocate(size_t size) {
    void * ptr = this->allocator.allocate(size);
    this->ptr_mem_table.insert({ptr, size});
    this->memory_usage += size;
  }
  void deallocate(void * ptr) {
    auto itr = this->ptr_mem_table.find(ptr);
    assert (itr != this->ptr_mem_table.end());
    this->memory_usage -= itr->second;
    this->allocator.deallocate(ptr);
    this->ptr_mem_table.erase(itr);
  }

  Allocator allocator;
  size_t memory_usage;

private:
  std::unordered_map<void*, size_t> ptr_mem_table;
};

} // namespace FlexFlow

#endif
