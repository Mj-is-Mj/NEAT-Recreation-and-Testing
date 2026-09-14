### Compiler stuff

CXX = g++

CXX_FLAGS = 
DEPEND_PATH = $(abspath "./dependencies")
CXX_FLAGS := ${CXX_FLAGS} -L${DEPEND_PATH}/lib/ -I${DEPEND_PATH}/include/

# CXX_FLAGS = `pkg-config --cflags --libs sdl3`

### Directories

DIR_SRC   = src
DIR_BUILD = build

### Dependency files

GLOB_CPP = $(wildcard $(DIR_SRC)/lib/*.cpp)
GLOB_OBJ = $(patsubst $(DIR_SRC)/%.cpp, $(DIR_BUILD)/%.o, $(GLOB_CPP))


### Targets

TESTS  = $(patsubst $(DIR_SRC)/tests/%.cpp,%,$(wildcard $(DIR_SRC)/tests/*.cpp))

TEST_PRE = test_
TEST_TARGETS = $(patsubst %,$(TEST_PRE)%,$(TESTS))

all: $(TESTS)


### Build proccesses

# App build
$(DIR_BUILD)/%: $(DIR_SRC)/tests/%.cpp $(GLOB_OBJ)
	mkdir -p $(dir $@)
	$(CXX) $^ $(CXX_FLAGS) -o $@


# Dep build
$(DIR_BUILD)/%.o: $(DIR_SRC)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) -c $< $(CXX_FLAGS) -o $@

### Phonies

.PHONY: all clean $(TESTS)

clean:
	echo ${GLOB_CPP}
	rm -rf $(DIR_BUILD)

$(TESTS): %: $(DIR_BUILD)/%
	@:

$(TEST_TARGETS): $(TEST_PRE)%: $(DIR_BUILD)/%
	./$<
