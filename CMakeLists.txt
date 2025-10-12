cmake_minimum_required(VERSION 3.16)
project(MyProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB TEST_MISTRAL "test/test_mistral/*")

add_executable(main_exec main.cpp ${SRC})
add_executable(test_exec ${SRC} ${TEST_MISTRAL})