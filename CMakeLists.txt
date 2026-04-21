cmake_minimum_required(VERSION 3.21)
project(sembed VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_subdirectory(external)

set(PROJECT_INCLUDE_DIR src/include)

add_library(utils STATIC
    src/utils/randomUtils.cpp
    src/utils/inMemDataset.cpp
    src/utils/diskUtils.cpp
    src/utils/HDVector.cpp
    src/utils/Vamana.cpp
    src/utils/Graph.cpp
)
add_library(batch_stocastic_kmeans STATIC src/load/batch_stocastic_kmeans.cpp)

target_include_directories(utils PUBLIC ${PROJECT_INCLUDE_DIR})
target_include_directories(batch_stocastic_kmeans PUBLIC ${PROJECT_INCLUDE_DIR})

target_link_libraries(batch_stocastic_kmeans PUBLIC utils)

add_executable(sembed src/main/main.cpp)
target_include_directories(sembed PUBLIC ${PROJECT_INCLUDE_DIR})
target_link_libraries(sembed PUBLIC utils)

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

add_executable(Test
    tests/Test_binaryLoading.cpp
    tests/Test_hdVectorCore.cpp
    tests/Test_greedySearch.cpp
    tests/Test_dataset_api.cpp
    tests/Test_boundaryAndPersistence.cpp
    tests/Test_randomUtilsAndSearch.cpp
    tests/Test_componentRegressions.cpp
    tests/Test_robustnessAndIntegration.cpp
)
target_link_libraries(Test PUBLIC GTest::gtest_main utils batch_stocastic_kmeans)
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
