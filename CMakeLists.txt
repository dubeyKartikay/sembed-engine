cmake_minimum_required(VERSION 3.21)
project(sembed VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SEMBED_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(SEMBED_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(OpenMP REQUIRED)
find_package(Armadillo QUIET)

include(FetchContent)

add_subdirectory(external)

set(PROJECT_INCLUDE_DIR src/include)

macro(sembed_fetch_header_dependency dependency_name repository tag)
    FetchContent_Declare(
        ${dependency_name}
        GIT_REPOSITORY ${repository}
        GIT_TAG ${tag}
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR cmake-do-not-add
    )
    FetchContent_MakeAvailable(${dependency_name})
endmacro()

add_library(sembed_project_options INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(sembed_project_options INTERFACE -Wall -Wextra -Wpedantic)
    if(SEMBED_ENABLE_ASAN)
        target_compile_options(sembed_project_options INTERFACE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(sembed_project_options INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    endif()
    if(SEMBED_ENABLE_UBSAN)
        target_compile_options(sembed_project_options INTERFACE -fsanitize=undefined -fno-omit-frame-pointer)
        target_link_options(sembed_project_options INTERFACE -fsanitize=undefined -fno-omit-frame-pointer)
    endif()
endif()

if(ARMADILLO_FOUND)
    add_library(sembed_armadillo INTERFACE)
    target_include_directories(sembed_armadillo INTERFACE ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(sembed_armadillo INTERFACE ${ARMADILLO_LIBRARIES})
else()
    FetchContent_Declare(
        armadillo
        GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
        GIT_TAG 36b0cd37ec7f8598cd3705e5eae8b4ad01f69559 # 15.2.6
    )
    FetchContent_MakeAvailable(armadillo)
    add_library(sembed_armadillo INTERFACE)
    target_link_libraries(sembed_armadillo INTERFACE armadillo)
endif()

find_path(MLPACK_INCLUDE_DIR NAMES mlpack.hpp)
if(NOT MLPACK_INCLUDE_DIR)
    sembed_fetch_header_dependency(
        mlpack
        https://github.com/mlpack/mlpack.git
        4.7.0
    )
    set(MLPACK_INCLUDE_DIR "${mlpack_SOURCE_DIR}/src" CACHE PATH "mlpack include directory" FORCE)
endif()

find_path(ENSMALLEN_INCLUDE_DIR NAMES ensmallen.hpp)
if(NOT ENSMALLEN_INCLUDE_DIR)
    sembed_fetch_header_dependency(
        ensmallen
        https://github.com/mlpack/ensmallen.git
        2.22.2
    )
    set(ENSMALLEN_INCLUDE_DIR "${ensmallen_SOURCE_DIR}/include" CACHE PATH "ensmallen include directory" FORCE)
endif()

find_path(CEREAL_INCLUDE_DIR NAMES cereal/cereal.hpp)
if(NOT CEREAL_INCLUDE_DIR)
    sembed_fetch_header_dependency(
        cereal
        https://github.com/USCiLab/cereal.git
        v1.3.2
    )
    set(CEREAL_INCLUDE_DIR "${cereal_SOURCE_DIR}/include" CACHE PATH "cereal include directory" FORCE)
endif()

add_library(sembed_mlpack INTERFACE)
target_include_directories(sembed_mlpack SYSTEM INTERFACE
    ${MLPACK_INCLUDE_DIR}
    ${ENSMALLEN_INCLUDE_DIR}
    ${CEREAL_INCLUDE_DIR}
)
target_link_libraries(sembed_mlpack INTERFACE sembed_armadillo OpenMP::OpenMP_CXX)

add_library(utils STATIC
    src/utils/ArmaVector.cpp
    src/utils/randomUtils.cpp
    src/utils/flatDataset.cpp
    src/utils/HDVector.cpp
    src/utils/Vamana.cpp
    src/utils/Graph.cpp
    src/utils/diskAnn.cpp
)
add_library(benchmark_support STATIC src/bench/benchmark_harness.cpp)
add_library(batch_stochastic_kmeans STATIC src/load/batch_stochastic_kmeans.cpp)
add_library(cli_support STATIC src/main/cli_workflow.cpp)

target_include_directories(utils PUBLIC ${PROJECT_INCLUDE_DIR})
target_include_directories(benchmark_support PUBLIC ${PROJECT_INCLUDE_DIR})
target_include_directories(batch_stochastic_kmeans PUBLIC ${PROJECT_INCLUDE_DIR})
target_include_directories(cli_support PUBLIC ${PROJECT_INCLUDE_DIR})

target_link_libraries(utils PUBLIC sembed_project_options OpenMP::OpenMP_CXX sembed_armadillo sembed_mlpack)
target_link_libraries(benchmark_support PUBLIC utils nlohmann_json::nlohmann_json)
target_link_libraries(batch_stochastic_kmeans PUBLIC utils)
target_link_libraries(cli_support PUBLIC utils CLI11::CLI11 nlohmann_json::nlohmann_json)

add_executable(sembed src/main/main.cpp)
target_include_directories(sembed PUBLIC ${PROJECT_INCLUDE_DIR})
target_link_libraries(sembed PUBLIC cli_support)

add_executable(sembed_benchmark src/bench/main.cpp)
target_include_directories(sembed_benchmark PUBLIC ${PROJECT_INCLUDE_DIR})
target_link_libraries(sembed_benchmark PUBLIC benchmark_support CLI11::CLI11)

add_executable(sembed_diskann_build_benchmark
    src/bench/diskann_build_benchmark.cpp
)
target_include_directories(sembed_diskann_build_benchmark PUBLIC
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(sembed_diskann_build_benchmark PUBLIC utils)

enable_testing()

set(GLOVE_FIXTURE "${CMAKE_BINARY_DIR}/gvec.bin")
set(WORD2VEC_FIXTURE "${CMAKE_BINARY_DIR}/w2v.bin")


add_custom_command(
    OUTPUT "${GLOVE_FIXTURE}" "${WORD2VEC_FIXTURE}"
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/scripts/generate_embedding_fixtures.py"
            --output-dir "${CMAKE_BINARY_DIR}"
    DEPENDS
        "${CMAKE_SOURCE_DIR}/scripts/generate_embedding_fixtures.py"
        "${CMAKE_SOURCE_DIR}/convertUtils/convertGloveTXT.c"
        "${CMAKE_SOURCE_DIR}/convertUtils/convertWord2vectDS.c"
        "${CMAKE_SOURCE_DIR}/testdata/embeddings/glove/glove.6B.50d.subset.txt"
        "${CMAKE_SOURCE_DIR}/testdata/embeddings/word2vec/google-word2vec.subset.bin"
    COMMENT "Generating embedding fixtures"
    VERBATIM
)

add_custom_target(embedding_fixtures DEPENDS "${GLOVE_FIXTURE}" "${WORD2VEC_FIXTURE}")

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/Test_*.cpp")
add_executable(Test ${TEST_SOURCES})
target_link_libraries(Test PUBLIC
    GTest::gtest_main
    utils
    batch_stochastic_kmeans
    benchmark_support
    cli_support
)
target_compile_definitions(Test PRIVATE
    SEMBED_TEST_BINARY_DIR="${CMAKE_BINARY_DIR}"
)
add_dependencies(Test embedding_fixtures)

include(GoogleTest)
gtest_discover_tests(Test)

set(REPORT_DIR "${CMAKE_BINARY_DIR}/test-report")

function(add_report_target name comment)
    add_custom_target(${name}
        DEPENDS Test
        COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${CMAKE_BINARY_DIR}" --output-on-failure
                --output-junit "${REPORT_DIR}/junit.xml"
        ${ARGN}
        COMMENT "${comment}"
        USES_TERMINAL
        VERBATIM
    )
endfunction()

add_report_target(test_report "Running tests and generating HTML report")

add_custom_target(benchmark_smoke
    DEPENDS sembed_benchmark embedding_fixtures
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/scripts/run_benchmarks.py"
            --benchmark-binary "$<TARGET_FILE:sembed_benchmark>"
            --config "${CMAKE_SOURCE_DIR}/benchmarks/local_smoke.json"
            --build-dir "${CMAKE_BINARY_DIR}"
            --output "${CMAKE_BINARY_DIR}/benchmark-report/local-smoke.json"
    COMMENT "Running the local benchmark smoke profile"
    USES_TERMINAL
    VERBATIM
)
