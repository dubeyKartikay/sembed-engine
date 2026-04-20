cmake_minimum_required(VERSION 3.21)
project(sembed VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(external)

set(PROJECT_INCLUDE_DIR src/include)

add_library(utils STATIC
    src/utils/randomUtils.cpp
    src/utils/inMemDataset.cpp
    src/utils/diskUtils.cpp
    src/utils/HDVector.cpp
    src/utils/Vamana.cpp
    src/utils/Graph.cpp
    src/utils/Set.cpp
)
add_library(batch_stocastic_kmeans STATIC src/load/batch_stocastic_kmeans.cpp)
add_library(load_from_binary STATIC src/load/load_from_binary.cpp)

target_include_directories(utils PUBLIC ${PROJECT_INCLUDE_DIR})
target_include_directories(batch_stocastic_kmeans PUBLIC ${PROJECT_INCLUDE_DIR})
target_include_directories(load_from_binary PUBLIC ${PROJECT_INCLUDE_DIR})

target_link_libraries(batch_stocastic_kmeans PUBLIC utils)
target_link_libraries(load_from_binary PUBLIC batch_stocastic_kmeans)

add_executable(sembed src/main/main.cpp)
target_include_directories(sembed PUBLIC ${PROJECT_INCLUDE_DIR})
target_link_libraries(sembed PUBLIC load_from_binary)

enable_testing()

add_executable(Test
    tests/Test_diskUtils.cpp
    tests/Test_inMemory.cpp
    tests/Test_hdVectorCore.cpp
    tests/Test_greedySearch.cpp
    tests/Test_dataset_api.cpp
    tests/Test_boundaryAndPersistence.cpp
    tests/Test_randomUtilsAndSearch.cpp
    tests/Test_componentRegressions.cpp
    tests/Test_robustnessAndIntegration.cpp
)
target_link_libraries(Test PUBLIC GTest::gtest_main utils load_from_binary)

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
