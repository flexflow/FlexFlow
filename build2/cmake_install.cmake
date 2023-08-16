# Install script for directory: /home/ubuntu/repo-refactor

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/ubuntu/repo-refactor/build2/deps/json/cmake_install.cmake")
  include("/home/ubuntu/repo-refactor/build2/deps/optional/cmake_install.cmake")
  include("/home/ubuntu/repo-refactor/build2/deps/expected/cmake_install.cmake")
  include("/home/ubuntu/repo-refactor/build2/deps/spdlog/cmake_install.cmake")
  include("/home/ubuntu/repo-refactor/build2/deps/variant/cmake_install.cmake")
  include("/home/ubuntu/repo-refactor/build2/deps/doctest/cmake_install.cmake")
  include("/home/ubuntu/repo-refactor/build2/deps/fmt/cmake_install.cmake")
  include("/home/ubuntu/repo-refactor/build2/deps/legion/cmake_install.cmake")
  include("/home/ubuntu/repo-refactor/build2/deps/rapidcheck/cmake_install.cmake")
  include("/home/ubuntu/repo-refactor/build2/deps/invoke/cmake_install.cmake")
  include("/home/ubuntu/repo-refactor/build2/lib/cmake_install.cmake")
  include("/home/ubuntu/repo-refactor/build2/bin/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/ubuntu/repo-refactor/build2/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
