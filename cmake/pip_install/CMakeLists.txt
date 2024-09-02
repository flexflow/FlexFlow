# Use setup.py script to re-install the Python bindings library with the right library paths
if (FF_USE_PYTHON)
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import site, os; print([pkg for func in (site.getsitepackages(), site.getusersitepackages()) for pkg in ([func] if isinstance(func, str) else func) if os.access(pkg, os.W_OK)][0])" OUTPUT_VARIABLE PY_DEST OUTPUT_STRIP_TRAILING_WHITESPACE)  
    if(FF_BUILD_FROM_PYPI)
        cmake_path(SET CMAKE_SOURCE_DIR_ NORMALIZE ${CMAKE_CURRENT_SOURCE_DIR}/../../deps/legion)
        cmake_path(SET CMAKE_BUILD_DIR_ NORMALIZE ${Legion_BINARY_DIR}/runtime)
        cmake_path(SET CMAKE_INSTALL_PREFIX_ NORMALIZE ${PY_DEST}/../../..)
        cmake_path(SET WORKING_DIRECTORY_ NORMALIZE ${CMAKE_CURRENT_SOURCE_DIR}/../../deps/legion/bindings/python/)
        # CMAKE_CURRENT_SOURCE_DIR=/usr/FlexFlow/cmake/pip_install 
        # Legion_BINARY_DIR=/usr/FlexFlow/build/<something>/deps/legion
        # CMAKE_SOURCE_DIR_=/usr/FlexFlow/deps/legion
        # CMAKE_BUILD_DIR_: /usr/FlexFlow/build/<something>/deps/legion/runtime
        # CMAKE_INSTALL_PREFIX_: /opt/conda/ or /usr/local
        # WORKING_DIRECTORY_: /usr/FlexFlow/deps/legion/bindings/python/
        # PY_DEST: <path to python lib>/python3.11/site-packages
        message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
        message(STATUS "Legion_BINARY_DIR: ${Legion_BINARY_DIR}")
        message(STATUS "CMAKE_SOURCE_DIR_: ${CMAKE_SOURCE_DIR_}")
        message(STATUS "CMAKE_BUILD_DIR_: ${CMAKE_BUILD_DIR_}")
        message(STATUS "CMAKE_INSTALL_PREFIX_: ${CMAKE_INSTALL_PREFIX_}")
        message(STATUS "WORKING_DIRECTORY_: ${WORKING_DIRECTORY_}")
        message(STATUS "PY_DEST: ${PY_DEST}")
        install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E echo \"Editing path to Legion library using path: ${CMAKE_INSTALL_PREFIX_} \")")
        install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E env CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR_} CMAKE_BUILD_DIR=${CMAKE_BUILD_DIR_} CMAKE_INSTALL_PREFIX=${PY_DEST}/flexflow ${Python3_EXECUTABLE} setup.py install --prefix ${CMAKE_INSTALL_PREFIX_} ${Legion_PYTHON_EXTRA_INSTALL_ARGS} WORKING_DIRECTORY ${WORKING_DIRECTORY_} COMMAND_ECHO STDOUT COMMAND_ERROR_IS_FATAL ANY)")
    endif()
endif()
