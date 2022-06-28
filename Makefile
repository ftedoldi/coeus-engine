# author: Davide Paolillo
# Real-Time Graphics Programming - a.a. 2021/2022

# name of the file
FILENAME = Main

# Visual Studio compiler
CC = cl.exe

# Include path
IDIR = /I./Includes /I./Includes/CUDA /I./Math /I./CUDA /I./CUDA/Shared /I./Math/Vector /I./Math/Versor /I./Math/Point /I./Math/Rotation /I./Math/Matrix /I./Rendering /I./Rendering/ModelLoader /I./DataStructs /I./Test      

# compiler flags:
CCFLAGS  = /Od /Zi /EHsc /MT

# linker flags:
LFLAGS = /LIBPATH:./Includes/Libraries assimp-vc142-mt.lib zlib.lib cuda.lib cudart.lib glfw3dll.lib glm_static.lib

SOURCES = ./Includes/src/glad.c $(FILENAME).cpp $(shell powershell -file .getCPPFilePaths.ps1)

TARGET = ./Build/CoeusEngine.exe

.PHONY : all
all:
	$(CC) $(CCFLAGS) $(IDIR) $(SOURCES) /Fe:$(TARGET) /link $(LFLAGS)

.PHONY : clean
clean :
	del *.obj *.lib *.exp *.ilk *.pdb
