# coeus engine

CUDA-based rendering and physics engine.

# Import Models

In order to avoid the excessive growth of the repo we decided to load all the models on a Drive.
Here you can find all the models used: https://drive.google.com/drive/u/1/folders/12JN1tYfdxabtCbQ0thMZrkp1TML8EXkX
In order to download the Model folder to your local machine just run the 'downloadModels.ps1' powershell script.

# CUDA Compilation

In order to perform CUDA compilation run the powershell script provided inside the CUDA folder.
Once the .lib and the .dll files are generated you might add the flag -l{NameOfThe".lib"FileWithoutExtension} to Tasks.json args.

# Compilation

Just simply press Ctrl + Shift + B in order to compile and run the project.