#include "kernels/managed_ff_stream.h"

namespace FlexFlow {

ManagedFFStream::ManagedFFStream() : stream(new ffStream_t) {
  checkCUDA(cudaStreamCreate(stream));
}

ManagedFFStream::ManagedFFStream(ManagedFFStream &&other) noexcept
    : stream(std::exchange(other.stream, nullptr)) {}

ManagedFFStream &ManagedFFStream::operator=(ManagedFFStream &&other) noexcept {
  std::swap(this->stream, other.stream);
  return *this;
}

ManagedFFStream::~ManagedFFStream() {
  if (stream != nullptr) {
    checkCUDA(cudaStreamDestroy(*stream));
    delete stream;
  }
}

ffStream_t const &ManagedFFStream::raw_stream() const {
  return *stream;
}

} // namespace FlexFlow
