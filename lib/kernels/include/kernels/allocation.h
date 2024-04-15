#ifndef _FLEXFLOW_KERNELS_ALLOCATION_H
#define _FLEXFLOW_KERNELS_ALLOCATION_H

#include "accessor.h"
// #include "pcg/tensor.h"
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

  // GenericTensorAccessorW allocate(Tensor tensor) {
  //   void *ptr = this->i_allocator->allocate(tensor.get_volume());
  //   GenericTensorAccessorW tensor_backing = {
  //       tensor.data_type, tensor.get_shape(), ptr};
  //   return tensor_backing;
  // }

  // void deallocate(GenericTensorAccessorW tensor_backing) {
  //   this->i_allocator->deallocate(tensor_backing.ptr);
  // }

  void *allocate(size_t mem_size) {
    return this->i_allocator->allocate(mem_size);
  }

  void deallocate(void *ptr) {
    this->i_allocator->deallocate(ptr);
  }

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IAllocator, T>::value,
                                 Allocator>::type
      create(Args &&...args) {
    return Allocator(std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  Allocator(std::shared_ptr<IAllocator> ptr) : i_allocator(ptr){};
  std::shared_ptr<IAllocator> i_allocator;
};

} // namespace FlexFlow

#endif
