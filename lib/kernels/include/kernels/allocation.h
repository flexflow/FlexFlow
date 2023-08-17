#ifndef _FLEXFLOW_KERNELS_ALLOCATION_H
#define _FLEXFLOW_KERNELS_ALLOCATION_H

#include <cstddef>
#include <memory>

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

} // namespace FlexFlow

#endif
