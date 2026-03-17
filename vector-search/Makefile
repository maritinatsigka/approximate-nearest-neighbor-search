CXX := g++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra -Wpedantic
INCLUDES := -Iinclude

SRC := src/dataset.cpp \
       src/hypercube.cpp \
       src/LSH.cpp \
       src/utils.cpp \
       src/main.cpp \
       src/ivfflat.cpp \
       src/ivfpq.cpp

OBJ := $(SRC:.cpp=.o)
BIN := search

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) $(OBJ) $(BIN)

.PHONY: all clean
