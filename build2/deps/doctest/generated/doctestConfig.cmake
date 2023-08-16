if(NOT TARGET doctest::doctest)
    # Provide path for scripts
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

    include("${CMAKE_CURRENT_LIST_DIR}/doctestTargets.cmake")
endif()
