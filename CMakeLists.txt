cmake_minimum_required(VERSION 3.20)

project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main source directory
set(SRC ${CMAKE_SOURCE_DIR}/src)
# Local dependencies directory
set(DEP ${CMAKE_SOURCE_DIR}/dep)

# Include directories
set(INCLUDE_DIRS
    ${SRC}/inc
    ${DEP}/include
)

# Common source files
file(GLOB SOURCES ${SRC}/lib/*.cpp)

# Make executables
function(make_exectuable EXE_SOURCE)
    get_filename_component(EXE_NAME ${EXE_SOURCE} NAME_WE)

    add_executable(${EXE_NAME}
        ${SOURCES}
        ${EXE_SOURCE}
    )

    target_include_directories(${EXE_NAME} PRIVATE
        ${INCLUDE_DIRS}
    )
endfunction()


# Get all test programs
file(GLOB TESTS src/tests/*.cpp)
# Make tests
foreach(TEST ${TESTS})
    make_exectuable(${TEST})
endforeach()