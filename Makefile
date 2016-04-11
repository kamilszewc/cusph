MONO = mcs
MAKE = make

all: sphectacle cusph2d cusph3d

sphectacle:
	$(MONO) Sphectacle-console/*.cs Sphectacle/*.cs -pkg:dotnet /out:sphectacle

cusph2d:
	$(MAKE) -C cuSPH2d 

cusph3d:
	$(MAKE) -C cuSPH3d
