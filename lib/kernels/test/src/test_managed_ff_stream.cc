#include "doctest/doctest.h"
#include "kernels/managed_ff_stream.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Managed FF Stream") {
    ManagedFFStream base_stream{};

    SUBCASE("Test ManagedFFStream Move Constructor") {
      ffStream_t const *base_stream_ptr = &base_stream.raw_stream();

      ManagedFFStream new_stream(std::move(base_stream));

      CHECK(&base_stream.raw_stream() == nullptr);
      CHECK(&new_stream.raw_stream() == base_stream_ptr);
    }

    SUBCASE("Test ManagedFFStream Assignment Operator") {
      ffStream_t const *base_stream_ptr = &base_stream.raw_stream();

      ManagedFFStream new_stream{};
      new_stream = std::move(base_stream);

      CHECK(&base_stream.raw_stream() == nullptr);
      CHECK(&new_stream.raw_stream() == base_stream_ptr);
    }
  }
}
