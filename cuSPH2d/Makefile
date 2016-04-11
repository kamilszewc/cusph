CXX = g++
CUARCH=sm_35
CU = nvcc
CXXFLAGS = -O2 -march=native -std=c++11
CUFLAGS = -O2 -arch=$(CUARCH)
RM = rm -f

ifdef debug
	CXXFLAGS += -g
	CUFLAGS += -g
endif

all: cusph

cusph: $(patsubst %.cpp, %.o, $(wildcard *.cpp)) \
       $(patsubst %.cpp, %.o, $(wildcard tinyxml2/*.cpp)) \
       $(patsubst %.cu, %.o, $(wildcard methods/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcDispersedPhase/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/renormalizePressure/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcStrainTensor/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcHydrostaticPressure/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/smoothingDensity/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcNumberOfCells/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcChezyViscosity/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcTimeStep/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/smoothHydrostaticPressure/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/dispersedPhaseFluidParticleManager/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/general/calcTimeStep/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/wcsphStandard/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/wcsphColagrossiLandrini/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/wcsphHuAdams/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/wcsphSzewcOlejnik/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/sphTartakovskyMeakin/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/sphTartakovskyEtAl/*.cu)) \
       $(patsubst %.cu, %.o, $(wildcard models/wcsphStandardDispersedPhase/*.cu))
	$(CU) $(CUFLAGS) $^ -dlink -o link.o 
	$(CXX) $^ link.o -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -o cusph
	
old:
	$(CXX) $(CXXFLAGS) -I/usr/local/cuda/include -c device.cpp domain.cpp input.cpp output.cpp terminal.cpp license.cpp errlog.cpp tinyxml2/tinyxml2.cpp
	$(CU) -O2 -arch=$(CUARCH) -std=c++11 models/*.cu models/general/calcChezyViscosity/*.cu models/general/calcNumberOfCells/*.cu models/general/calcDispersedPhase/*.cu models/general/renormalizePressure/*.cu models/general/calcStrainTensor/*.cu models/general/calcHydrostaticPressure/*.cu models/general/smoothHydrostaticPressure/*.cu models/general/smoothingDensity/*.cu models/general/calcTimeStep/*.cu models/wcsphStandard/*.cu models/wcsphStandardDispersedPhase/*.cu models/wcsphColagrossiLandrini/*.cu models/sphTartakovskyEtAl/*.cu models/wcsphHuAdams/*.cu models/wcsphSzewcOlejnik/*.cu models/sphTartakovskyMeakin/*.cu models/general/dispersedPhaseFluidParticleManager/*.cpp methods/*.cu *.o sph.cpp -o cusph

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -I/usr/local/cuda/include $< -o $@

%.o: %.cu
	$(CU) $(CUFLAGS) -dc $< -o $@	

clean:
	$(RM) *.o methods/*.o models/*.o models/general/calcDispersedPhase/*.o models/general/renormalizePressure/*.o models/general/calcChezyViscosity/*.o models/general/calcNumberOfCells/*.o models/general/calcHydrostaticPressure/*.o models/general/smoothHydrostaticPressure/*.o models/general/calcStrainTensor/*.o models/general/calcTimeStep/*.o models/general/smoothingDensity/*.o models/wcsphStandard/*.o models/wcsphStandardDispersedPhase/*.o models/wcsphColagrossiLandrini/*.o models/wcsphHuAdams/*.o models/wcsphSzewcOlejnik/*.o models/sphTartakovskyMeakin/*.o models/sphTartakovskyEtAl/*.o odels/general/dispersedPhaseFluidParticleManager/*.o tinyxml2/*.o cusph postproc/*.pyc postproc/postproc/*.pyc
mrproper: clean
	$(RM) results/* logs/*
