#include "kernels/managed_ff_stream.h"

namespace FlexFlow {
ManagedFFStream::ManagedFFStream() {
  checkCUDA(cudaStreamCreate(&stream));
}

ManagedFFStream::~ManagedFFStream() {
  checkCUDA(cudaStreamDestroy(stream));
}

} // namespace FlexFlow
