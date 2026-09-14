cmake_minimum_required(VERSION 3.20)

project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Common source files
file(GLOB SOURCES src/lib/*.cpp)

# Get all test executables
file(GLOB TESTS src/tests/*.cpp)

# Make test executables
foreach(TEST ${TESTS})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)

    add_executable(${TEST_NAME}
        ${SOURCES}
        ${TEST}
    )

    target_include_directories(${TEST_NAME} PRIVATE
        src/inc
    )
endforeach()