# author: Davide Paolillo
# Real-Time Graphics Programming - a.a. 2021/2022

# name of the file
FILENAME = Main

# Visual Studio compiler
CC = cl.exe

# Include path
IDIR = /I./Includes /I./Includes/CUDA /I./Math /I./CUDA /I./CUDA/Shared /I./Math/Vector /I./Math/Versor /I./Math/Point /I./Rendering/Scene /I./Math/Rotation /I./Math/Matrix /I./Rendering /I./Rendering/Texture /I./Rendering/ModelLoader /I./DataStructs /I./Test /I./System /I./Includes/imgui /I./Rendering/Light     

# compiler flags:
CCFLAGS  = /Od /Zi /EHsc /MT /std:c++17

# linker flags:
LFLAGS = ./Resources/resource.res /LIBPATH:./Includes/Libraries assimp-vc142-mt.lib zlib.lib cuda.lib cudart.lib glfw3dll.lib glm_static.lib

SOURCES = ./Includes/src/glad.c $(FILENAME).cpp $(shell powershell -file .getCPPFilePaths.ps1)
PARTIAL_SOURCES = $(shell python ./findAllObjs.py $(ARGS))

TARGET = ./Build/CoeusEngine.exe

.PHONY : all
all:
	$(CC) $(CCFLAGS) $(IDIR) $(SOURCES) /Fe:$(TARGET) /link $(LFLAGS)

.PHONY : partial
partial:
	$(CC) $(CCFLAGS) $(IDIR) $(PARTIAL_SOURCES) /Fe:$(TARGET) /link $(LFLAGS)

.PHONY : clean
clean :
	del *.obj *.lib *.exp *.ilk *.pdb
