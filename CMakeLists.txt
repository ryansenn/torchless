cmake_minimum_required(VERSION 3.16)
project(MyProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE TEST_MISTRAL CONFIGURE_DEPENDS "test/mistral/*.cpp")

add_executable(main_exec main.cpp ${SRC})
add_executable(test_exec ${SRC} ${TEST_MISTRAL})
configure_file(test/mistral/expected.txt expected.txt COPYONLY)