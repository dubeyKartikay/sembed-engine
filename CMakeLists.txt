cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external)
project(sembed VERSION 1.0)
add_executable(sembed ${CMAKE_CURRENT_SOURCE_DIR}/src/main/main.cpp )
include_directories(sembed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
add_library(load_from_binary STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/load/load_from_binary.cpp)
add_library(batch_stocastic_kmeans STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/load/batch_stocastic_kmeans.cpp)

set(UTILS_SRC_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/randomUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/inMemDataset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/diskUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/HDVector.cpp

    )

add_library(utils STATIC ${UTILS_SRC_FILES})

target_link_libraries(sembed PUBLIC gtest) 
target_link_libraries(batch_stocastic_kmeans PUBLIC utils)
target_link_libraries(load_from_binary PUBLIC batch_stocastic_kmeans)
target_link_libraries(sembed PUBLIC load_from_binary)

enable_testing()
set(TEST_SRC_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/Test_diskUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/Test_inMemory.cpp

)

add_executable(Test ${TEST_SRC_FILES}) 
target_link_libraries(Test PUBLIC GTest::gtest_main) 
target_link_libraries(Test PUBLIC utils)
include(GoogleTest)
gtest_discover_tests(Test)
