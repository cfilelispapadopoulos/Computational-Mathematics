# Compilers
CC = g++

# Object files go into $(OBJDIR), executables go into $(DISTDIR)
OBJDIR = obj
DISTDIR = build
SOURCEDIR = src
INCDIR = inc

# Compiler flags
# -fPIC is essential for creating a shared library (.so)
CXXFLAGS := -Wall -fPIC -O3 -std=c++17 -fopenmp -I$(INCDIR) -I$(MKLROOT)/include
FFLAGS   := -O3 -fPIC -Wno-tabs# -Wno-tabs helps with older Fortran source

# Linker flags
# Added -lgfortran to support the Fortran objects
MKL_LIBS := -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
LFLAGS   := $(CXXFLAGS) $(MKL_LIBS)

# File Discovery
HPPFILES := $(wildcard $(SOURCEDIR)/*.hpp)
CXXFILES := $(wildcard $(SOURCEDIR)/*.cpp)

# Object file mapping
# This puts all .o files into the obj directory regardless of source type
OFILES := $(patsubst $(SOURCEDIR)/%.cpp,$(OBJDIR)/%.o,$(CXXFILES)) \
          $(patsubst $(SOURCEDIR)/%.f,$(OBJDIR)/%.o,$(FFILES))

          .PHONY: all clean

          all: $(OBJDIR) $(DISTDIR) $(OFILES)
          	# Build Shared Library
		$(CC) -shared -o $(DISTDIR)/libafiim.so $(OFILES) $(LFLAGS)
	
		# Build Static Library
	 	ar rcs $(DISTDIR)/libafiim.a $(OFILES)
	 		
		# Build Example
		$(CC) $(CXXFLAGS) -I$(SOURCEDIR) example/example.cpp -o example/example.out \
		    -Wl,--whole-archive $(DISTDIR)/libafiim.a -Wl,--no-whole-archive $(LFLAGS)
	
# Rule for C++ files
$(OBJDIR)/%.o : $(SOURCEDIR)/%.cpp $(HPPFILES)
	$(CC) $(CXXFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(DISTDIR):
	mkdir -p $(DISTDIR)

clean:
	rm -rf $(OBJDIR) $(DISTDIR)
	rm -rf example/example.out
	rm -rf example/solution.mtx
