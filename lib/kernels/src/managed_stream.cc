#include "kernels/managed_stream.h"

namespace FlexFlow {
ManagedStream::ManagedStream() {
  checkCUDA(cudaStreamCreate(&stream));
}

ManagedStream::~ManagedStream() {
  checkCUDA(cudaStreamDestroy(stream));
}

ManagedStream get_managed_stream() {
  return ManagedStream();
}
} // namespace FlexFlow
