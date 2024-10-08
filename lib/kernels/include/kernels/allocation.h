#ifndef _FLEXFLOW_KERNELS_ALLOCATION_H
#define _FLEXFLOW_KERNELS_ALLOCATION_H

#include "accessor.h"
#include <cstddef>
#include <memory>

namespace FlexFlow {

struct IAllocator {
  virtual void *allocate(size_t) = 0;
  virtual void deallocate(void *) = 0;

  virtual DeviceType get_allocation_device_type() const = 0;

  virtual ~IAllocator() = default;
};

struct Allocator {
  Allocator() = delete;

  GenericTensorAccessorW allocate_tensor(TensorShape const &tensor_shape);

  void *allocate(size_t mem_size);
  void deallocate(void *ptr);

  DeviceType get_allocation_device_type() const;

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IAllocator, T>::value,
                                 Allocator>::type
      create(Args &&...args) {
    return Allocator(std::make_shared<T>(std::forward<Args>(args)...));
  }

  Allocator(std::shared_ptr<IAllocator> ptr) : i_allocator(ptr){};

private:
  std::shared_ptr<IAllocator> i_allocator;
};

} // namespace FlexFlow

#endif
