#!/usr/bin/make

# Name of generated binary
TARGET=cl_test
# All your source codes to build the target
SRCS=main.cpp
# Automatic generate other deps
OBJS=$(addprefix objs/, $(patsubst %.cpp,%.o,$(SRCS)))
DEPS=$(addprefix objs/, $(patsubst %.cpp,%.d,$(SRCS)))

CFLAGS=-g -Wall -O2
LFLAGS=-lOpenCL
EXTRA_CFLAGS=-Wno-deprecated-declarations

.PHONY: all clean execute directories

all : directories $(TARGET)


execute : all
	./$(TARGET) 

clean :
	rm -f objs/* $(TARGET)

directories :
	@mkdir -p objs

$(TARGET) : $(OBJS)
	@echo "  LINK  $@"
	@$(CXX) $^ -o $@ $(CFLAGS) $(LFLAGS) $(EXTRA_CFLAGS)

objs/%.o : %.cpp
	@echo "  CXX    $@"
	@$(CXX) -c $< -o $@ $(CFLAGS) $(EXTRA_CFLAGS)

objs/%.d : %.cpp
	@$(CXX) -MM $^ -MT $(patsubst %.d,%.o,$@) > $@

# Include automatically generated dependency files
-include $(DEPS)