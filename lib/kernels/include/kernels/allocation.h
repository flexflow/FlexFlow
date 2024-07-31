#ifndef _FLEXFLOW_KERNELS_ALLOCATION_H
#define _FLEXFLOW_KERNELS_ALLOCATION_H

#include "accessor.h"
#include <cstddef>
#include <memory>

enum class AllocLocation { HOST, DEVICE };

namespace FlexFlow {

struct IAllocator {
  virtual void *allocate(size_t) = 0;
  virtual void *allocate_and_zero(size_t) = 0;
  virtual void deallocate(void *) = 0;

  virtual ~IAllocator() = default;
};

struct Allocator {
  Allocator() = delete;

  GenericTensorAccessorW allocate_tensor(TensorShape const &tensor_shape);
  GenericTensorAccessorW
      allocate_tensor_and_zero(TensorShape const &tensor_shape);

  void *allocate(size_t mem_size);
  void *allocate_and_zero(size_t mem_size);
  void deallocate(void *ptr);

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IAllocator, T>::value,
                                 Allocator>::type
      create(Args &&...args) {
    return Allocator(std::make_shared<T>(std::forward<Args>(args)...));
  }

  Allocator(std::shared_ptr<IAllocator> ptr) : i_allocator(ptr){};

  AllocLocation alloc_location;

private:
  std::shared_ptr<IAllocator> i_allocator;
};

} // namespace FlexFlow

#endif
