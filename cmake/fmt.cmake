include(aliasing)

if (FF_USE_EXTERNAL_FMT)
  find_package(fmt REQUIRED)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/fmt)
endif()
alias_library(fmt fmt::fmt)
