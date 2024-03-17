if (FF_USE_EXTERNAL_RAPIDCHECK)
  find_package(rapidcheck REQUIRED)
  # find_package(PkgConfig REQUIRED)
  # pkg_search_module(RAPIDCHECK REQUIRED rapidcheck_doctest)

  # # add_library(rapidcheck INTERFACE)
  # set_target_properties(rapidcheck PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${RAPIDCHECK_INCLUDE_DIRS})
  # target_link_libraries(rapidcheck INTERFACE ${RAPIDCHECK_LIBRARIES})
  # target_compile_options(rapidcheck INTERFACE ${RAPIDCHECK_CFLAGS_OTHER})
# set_target_properties
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/rapidcheck)
endif()
