#include ./config.mak -> you can move all the FLAG to here and make file without config.mak file
#get all c++ files.
SRCS=$(wildcard *.c)
OBJS=$(addprefix objs/, $(patsubst %.c,%.o,$(SRCS)))
DEPS=$(addprefix objs/, $(patsubst %.c,%.d,$(SRCS)))

CC=gcc

CFLAGS=-g -Wall -std=c99 -ffast-math#`pkg-config --cflags opencv` 
LIBS = #`pkg-config --libs opencv`

# Header file: cl.h, Library: libOpenCL.so
# header file has already included in cl.h
PREFIX=/usr/lib/x86_64-linux-gnu/
LFLAGS=-I$(PREFIX)/ -L$(PREFIX)/ -lOpenCL 

LOCAL_CFLAGS=-Wno-deprecated-declarations

TARGET=mmult

.PHONY: all clean execute directories

all	:	directories $(TARGET)

$(TARGET)	:	$(OBJS)
	@echo "  LINK  $@"
	@$(CXX) $^ -o $@ $(CFLAGS) $(LIBS) $(LFLAGS) $(LOCAL_CFLAGS)

objs/%.o	:	%.c
	@echo "  CC    $@"
	@$(CXX) -c $< -o $@ $(CFLAGS) $(LOCAL_CFLAGS)

objs/%.d	:	%.c
	@$(CXX) -MM $^ -MT $(patsubst %.d,%.o,$@) > $@

directories	:	
	@mkdir -p objs

clean	:	
	rm -f objs/* $(TARGET)

execute	:	all
	./$(TARGET)

# Include automatically generated dependency files
-include $(DEPS)