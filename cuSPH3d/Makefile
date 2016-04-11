CXX = g++
CU = nvcc
CUARCH = sm_35
CXXFLAGS = -O2 -march=native -std=c++11
CUFLAGS = -O2 -arch=$(CUARCH)
RM = rm -f

ifdef debug
	CXXFLAGS += -g
	CUFLAGS += -g
endif

ifdef profile
	CUFLAGS += --ptxas-options=-v
endif

all: cusph

cusph: $(patsubst %.cpp, %.o, $(wildcard *.cpp)) \
       $(patsubst %.cpp, %.o, $(wildcard tinyxml2/*.cpp)) \
       $(patsubst %.cu, %.o, $(wildcard methods/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcDispersedPhase/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcShearRate/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcSingleSolidParticleAcceleration/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcTurbulentViscosity/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/smoothingDensity/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/wcsphStandard/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/wcsphColagrossiLandrini/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/wcsphHuAdams/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/wcsphSzewcOlejnik/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcTimeStep/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/renormalizePressure/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcHydrostaticPressure/*.cu))
	$(CU) $(CUFLAGS) $^ -dlink -o link.o 
	$(CXX) $^ link.o -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -o cusph
	
old:
	$(CXX) $(CXXFLAGS) -I/usr/local/cuda/include -c device.cpp domain.cpp input.cpp output.cpp terminal.cpp license.cpp errlog.cpp tinyxml2/tinyxml2.cpp
	$(CU) -O2 -arch=$(CUARCH) -std=c++11 models/*.cu models/general/*.cu models/general/smoothingDensity/*.cu models/general/calcTimeStep/*.cu models/general/calcHydrostaticPressure/*.cu models/general/renormalizePressure/*.cu models/general/calcDispersedPhase/*.cu models/general/calcShearRate/*.cu models/general/calcSingleSolidParticleAcceleration/*.cu models/general/calcTurbulentViscosity/*.cu models/wcsphStandard/*.cu models/wcsphColagrossiLandrini/*.cu models/wcsphHuAdams/*.cu models/wcsphSzewcOlejnik/*.cu methods/*.cu *.o sph.cpp -o cusph

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -I/usr/local/cuda/include $< -o $@

%.o: %.cu
	$(CU) $(CUFLAGS) -dc $< -o $@	

clean:
	$(RM) *.o methods/*.o models/*.o models/general/*.o models/general/calcTimeStep/*.o models/general/calcHydrostaticPressure/*.o models/general/smoothingDensity/*.o models/general/renormalizePressure/*.o models/general/calcDispersedPhase/*.o models/general/calcShearRate/*.o models/general/calcSingleSolidParticleAcceleration/*.o models/general/calcTurbulentViscosity/*.o models/wcsphStandard/*.o models/wcsphColagrossiLandrini/*.o models/wcsphHuAdams/*.o models/wcsphSzewcOlejnik/*.o tinyxml2/*.o cusph postproc/*.pyc postproc/postproc/*.pyc
mrproper: clean
	$(RM) results/* logs/*
