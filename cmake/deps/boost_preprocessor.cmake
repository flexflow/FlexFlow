include(aliasing)

if (FF_USE_EXTERNAL_BOOST_PREPROCESSOR)
  find_package(Boost REQUIRED)
  alias_library(boost_preprocessor Boost::boost)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/boost/preprocessor)
endif()
