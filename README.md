# coeus engine

CUDA-based rendering and physics engine.

# Import Models

In order to avoid the excessive growth of the repo we decided to load all the models on a Drive. <br />
Here you can find all the models used: https://drive.google.com/drive/u/1/folders/12JN1tYfdxabtCbQ0thMZrkp1TML8EXkX <br />
In order to download the Model folder to your local machine just run the 'downloadModels.ps1' powershell script.

# Load Models

To load a model with textures, make sure that in the same directory as the model, there is a folder called "Textures" which contains all used textures.
If model's textures are somewhere else, a full black model will be loaded.

# CUDA Compilation

In order to perform CUDA compilation run the powershell script provided inside the CUDA folder. <br />
Once the .lib and the .dll files are generated you might add the flag -l{NameOfThe".lib"FileWithoutExtension} to Tasks.json args.

# Compilation

Just simply press Ctrl + Shift + B in order to fully (re)compile and run the project.
If you wish to compile only some components in order to avoid the full recompilation use the script <br />
PartialMakefile.bat with subsequent parameters -> ex.: PartialMakefile.bat camera model <br />
this command will recompile only Camera.cpp and Model.cpp classes resulting in a fast paced compilation.
