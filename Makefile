CXX = g++
CXXFLAGS = -std=c++17 -O2

SRC = infer.cpp model/model.cpp
HEADERS = model/model.h

all: main test

main: main.cpp $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o main_exec main.cpp $(SRC)

test: test.cpp $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o test_exec test.cpp $(SRC)

clean:
	rm -f main_exec test_exec