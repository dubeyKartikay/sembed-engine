cmake_minimum_required(VERSION 3.21)
project(sembed VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SEMBED_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(SEMBED_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_subdirectory(external)

set(PROJECT_INCLUDE_DIR src/include)

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

add_library(utils STATIC
    src/utils/randomUtils.cpp
    src/utils/inMemDataset.cpp
    src/utils/diskUtils.cpp
    src/utils/HDVector.cpp
    src/utils/Vamana.cpp
    src/utils/Graph.cpp
)
add_library(benchmark_support STATIC src/bench/benchmark_harness.cpp)
add_library(batch_stochastic_kmeans STATIC src/load/batch_stochastic_kmeans.cpp)
add_library(cli_support STATIC src/main/cli_workflow.cpp)

target_include_directories(utils PUBLIC ${PROJECT_INCLUDE_DIR})
target_include_directories(benchmark_support PUBLIC ${PROJECT_INCLUDE_DIR})
target_include_directories(batch_stochastic_kmeans PUBLIC ${PROJECT_INCLUDE_DIR})
target_include_directories(cli_support PUBLIC ${PROJECT_INCLUDE_DIR})

target_link_libraries(utils PUBLIC sembed_project_options)
target_link_libraries(benchmark_support PUBLIC utils nlohmann_json::nlohmann_json)
target_link_libraries(batch_stochastic_kmeans PUBLIC utils)
target_link_libraries(cli_support PUBLIC utils CLI11::CLI11 nlohmann_json::nlohmann_json)

add_executable(sembed src/main/main.cpp)
target_include_directories(sembed PUBLIC ${PROJECT_INCLUDE_DIR})
target_link_libraries(sembed PUBLIC cli_support)

add_executable(sembed_benchmark src/bench/main.cpp)
target_include_directories(sembed_benchmark PUBLIC ${PROJECT_INCLUDE_DIR})
target_link_libraries(sembed_benchmark PUBLIC benchmark_support CLI11::CLI11)

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
