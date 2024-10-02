# This is for C but can easily be modified to work for other languages as well
CC = g++

# Object files go into $(OBJDIR), executables go into $DISTDIR
OBJDIR = obj
DISTDIR = build
SOURCEDIR = src
INCDIR = inc

# Compiler and linker flags
CXXFLAGS := -Wall -fPIC -O3
LFLAGS := $(CXXFLAGS) -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -ldl -lpthread -lm

# Gathers all header and C files located in the root directory
# in $(HFILES) and $(CFILES), respectively
HPPFILES := $(wildcard $(SOURCEDIR)/*.hpp)
CXXFILES := $(wildcard $(SOURCEDIR)/*.cpp)

# Produces in $(OFILES) the names of .o object files for all C files
OFILES := $(patsubst %.cpp,%.o,$(CXXFILES))

# This a rule that defines how to compile all C files that have changed
# (or their header files) since they were compiled last
$(OBJDIR)/%.o : %.cpp $(HPPFILES) $(CXXFILES)
	$(CC) $(CXXFLAGS) -c $< -o $@

# Consider these targets as targets, not files
.PHONY : all clean

# Build everything: compile all C (changed) files and
# link the object files into an executable (app)
all: $(OFILES)
	mkdir -p $(DISTDIR)
	$(CC) $(LFLAGS) $(OFILES) -shared -o $(DISTDIR)/libafiim.so
	mv $(SOURCEDIR)/*.o $(OBJDIR)
	ar rcs $(DISTDIR)/libafiim.a $(OBJDIR)/*.o
	$(CC) $(LFLAGS) -I$(SOURCEDIR) example/example.cpp $(DISTDIR)/libafiim.a -o example/example.out

# $(OBJDIR) is an order-only prerequisite:
# only compile if $(OBJDIR) does not exist yet and
# not just because $(OBJDIR) has a new time stamp
$(OFILES): | $(OBJDIR)

# Create directory for $(OBJDIR)
$(OBJDIR):
	mkdir $(OBJDIR)

# Clean up by removing the $(OBJDIR) and $(DISTDIR) directories
clean:
	rm -R $(OBJDIR)
	rm -R $(DISTDIR)
	rm -rf example/solution.mtx
	rm -rf example/example.out
