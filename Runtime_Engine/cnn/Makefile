ifneq ($(RESNET50),)
CPPFLAGS += -DRESNET50
endif

ifneq ($(RESNET50_PRUNED),)
CPPFLAGS += -DRESNET50_PRUNED
endif

ifeq ($(VERBOSE),1)
ECHO :=
else
ECHO := @
endif

# Compilation flags
ifeq ($(DEBUG),1)
CXXFLAGS += -g
else
CXXFLAGS += -O2
endif

# Compiler
CXX := g++ -fPIC -std=c++11

# Target
TARGET := tf2
TARGET_DIR := bin

# Directories
INC_DIRS := ../common/inc ./host/inc/
LIB_DIRS :=

# Files
INCS := $(wildcard)
SRCS := $(wildcard host/src/*.cpp)
LIBS := rt pthread opencv_core opencv_imgproc opencv_highgui opencv_ml opencv_imgcodecs coral-api

# Make it all!
all : $(TARGET_DIR)/$(TARGET)

# Host executable target.
$(TARGET_DIR)/$(TARGET) : Makefile $(SRCS) $(INCS) $(TARGET_DIR)
	$(ECHO)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(SRCS) \
			$(foreach D,$(LIB_DIRS),-L$D) \
			$(foreach L,$(LIBS),-l$L) -fopenmp \
			-o $(TARGET_DIR)/$(TARGET)

$(TARGET_DIR) :
	$(ECHO)mkdir $(TARGET_DIR)

# Standard make targets
clean :
	$(ECHO)rm -f $(TARGET_DIR)/$(TARGET)

.PHONY : all clean
