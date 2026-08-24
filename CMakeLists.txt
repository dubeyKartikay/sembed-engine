cmake_minimum_required(VERSION 3.21)
project(sembed VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(SEMBED_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(SEMBED_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

set(SEMBED_PUBLIC_HEADERS
    include/sembed/dataset_view.hpp
    include/sembed/index.hpp
    include/sembed/query.hpp
    include/sembed/sembed.hpp
    include/sembed/vector_view.hpp
)

add_library(sembed-lib STATIC
    ${SEMBED_PUBLIC_HEADERS}
    src/lib/sembed.cpp
    src/utils/HDVector.cpp
    src/utils/randomUtils.cpp
    src/utils/flatDataset.cpp
    src/utils/Vamana.cpp
    src/utils/Graph.cpp
)
add_library(sembed::sembed ALIAS sembed-lib)

set_target_properties(sembed-lib PROPERTIES
    OUTPUT_NAME sembed
    EXPORT_NAME sembed
    PUBLIC_HEADER "${SEMBED_PUBLIC_HEADERS}"
)

target_compile_features(sembed-lib PUBLIC cxx_std_17)
target_include_directories(sembed-lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(sembed-lib PRIVATE -Wall -Wextra -Wpedantic)
    if(SEMBED_ENABLE_ASAN)
        target_compile_options(sembed-lib PUBLIC -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(sembed-lib PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    endif()
    if(SEMBED_ENABLE_UBSAN)
        target_compile_options(sembed-lib PUBLIC -fsanitize=undefined -fno-omit-frame-pointer)
        target_link_options(sembed-lib PUBLIC -fsanitize=undefined -fno-omit-frame-pointer)
    endif()
endif()

add_executable(sembed
    src/main/main.cpp
    src/main/cli_workflow.cpp
)
target_compile_features(sembed PRIVATE cxx_std_17)
target_include_directories(sembed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
target_include_directories(sembed SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/CLI11/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/json/single_include
)
target_link_libraries(sembed PRIVATE sembed-lib)

install(TARGETS sembed-lib
    EXPORT sembedTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sembed
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(TARGETS sembed RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(EXPORT sembedTargets
    FILE sembedTargets.cmake
    NAMESPACE sembed::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sembed
)

configure_package_config_file(
    cmake/sembedConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sembedConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sembed
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sembedConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sembedConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/sembedConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sembed
)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
enable_testing()

set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory(external/googletest EXCLUDE_FROM_ALL)

file(GLOB SEMBED_TEST_SOURCES CONFIGURE_DEPENDS tests/Test_*.cpp)
add_executable(sembed-tests
    ${SEMBED_TEST_SOURCES}
    src/main/cli_workflow.cpp
)
target_compile_features(sembed-tests PRIVATE cxx_std_17)
target_include_directories(sembed-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
target_include_directories(sembed-tests SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/CLI11/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/json/single_include
)
target_link_libraries(sembed-tests PRIVATE sembed-lib GTest::gtest_main)
add_dependencies(sembed-tests sembed)
target_compile_definitions(sembed-tests PRIVATE
    SEMBED_TEST_BINARY_DIR="${CMAKE_BINARY_DIR}"
    SEMBED_TEST_PYTHON="${Python3_EXECUTABLE}"
    SEMBED_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_command(TARGET sembed-tests PRE_BUILD
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_embedding_fixtures.py
            --output-dir ${CMAKE_BINARY_DIR}
    COMMENT "Generating embedding fixtures for native tests"
    VERBATIM
)

include(GoogleTest)
gtest_discover_tests(sembed-tests)

add_test(
    NAME sembed_python_integration
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sembed.py
            --sembed $<TARGET_FILE:sembed>
)
