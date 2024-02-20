#ifndef _FLEXFLOW_KERNELS_ALLOCATION_H
#define _FLEXFLOW_KERNELS_ALLOCATION_H

#include <cstddef>
#include <memory>
#include "runtime/src/tensor.h"

namespace FlexFlow {

struct IAllocator {
  virtual void *allocate(Tensor) = 0;
  virtual void deallocate(void *) = 0;

  virtual ~IAllocator() = default;
};

struct Allocator {
  Allocator() = delete;

  void *allocate(Tensor);
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
