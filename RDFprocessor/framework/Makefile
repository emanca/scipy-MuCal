GCC=g++
CXXFLAGS=`root-config --libs --cflags` -O3 -fPIC -Wall -I./interface/  -I./ -I /scratchnvme/wmass/eigen-3.3.7/
SOFLAGS=-shared

SRCDIR=src
LIBDIR=lib
HPPDIR=interface
AUXDIR=aux

SRC=$(wildcard $(SRCDIR)/*.cpp)
OBJ=$(patsubst $(SRCDIR)/%.cpp, $(LIBDIR)/%.o , $(SRC)  )
HPPLINKDEF=$(patsubst $(SRCDIR)/%.cpp, ../interface/%.hpp , $(SRC)  )

.PHONY: all
all:
	$(info, "--- Full compilation --- ")	
	$(info, "-> if you want just to recompile something use 'make fast' ")	
	$(info, "------------------------ ")	
	$(MAKE) clean
	$(MAKE) libRDFProcessorCore.so

.PHONY: fast
fast:
	$(MAKE) libRDFProcessorCore.so

libRDFProcessorCore.so: $(OBJ) | $(LIBDIR)
	$(GCC) $(CXXFLAGS) $(RPATH) $(SOFLAGS) -o $(LIBDIR)/$@ $(OBJ) 

$(OBJ) : $(LIBDIR)/%.o : $(SRCDIR)/%.cpp interface/%.hpp | $(LIBDIR)
	$(GCC) $(CXXFLAGS) $(RPATH) -c -o $(LIBDIR)/$*.o $<

$(LIBDIR):
	mkdir -p $(LIBDIR)

.PHONY: clean
clean:
	-rm $(OBJ)
	-rm $(LIBDIR)/libRDFProcessorCore.so
	-rmdir $(LIBDIR)
