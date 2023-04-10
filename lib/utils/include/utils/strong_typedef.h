#ifndef _FLEXFLOW_UTILS_INCLUDE_STRONG_TYPEDEF_H
#define _FLEXFLOW_UTILS_INCLUDE_STRONG_TYPEDEF_H

#include <type_traits>
#include <functional>

namespace FlexFlow {

// derived from https://www.foonathan.net/2016/10/strong-typedefs/
template <typename Tag, typename T>
class strong_typedef {
public:
    strong_typedef() = delete;

    explicit strong_typedef(const T& value) 
      : value_(value)
    { }

    explicit strong_typedef(T&& value)
        noexcept(std::is_nothrow_move_constructible<T>::value)
      : value_(std::move(value))
    { }

    explicit operator T&() noexcept
    {
        return value_;
    }

    explicit operator const T&() const noexcept
    {
        return value_;
    }

    friend void swap(strong_typedef& a, strong_typedef& b) noexcept
    {
        using std::swap;
        swap(static_cast<T&>(a), static_cast<T&>(b));
    }

private:
    T value_;
};

}

namespace std {

template <typename Tag, typename T>
struct hash<::FlexFlow::strong_typedef<Tag, T>> {
  size_t operator()(::FlexFlow::strong_typedef<Tag, T> const &v) const {
    return get_std_hash((T const &)v);
  }
};

}

#endif
