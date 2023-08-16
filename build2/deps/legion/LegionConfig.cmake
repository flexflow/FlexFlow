list(INSERT CMAKE_MODULE_PATH 0 /home/ubuntu/repo-refactor/deps/legion/cmake)
include(${CMAKE_CURRENT_LIST_DIR}/LegionConfigCommon.cmake)

# Set our version variables
set(Legion_VERSION_MAJOR 22)
set(Legion_VERSION_MINOR 9)
set(Legion_VERSION_PATCH 0)
set(Legion_VERSION 22.9.0)
