import os
import shutil

cudaObjs = []

try:
    cudaObjs += [each for each in os.listdir("./CUDA/") if each.endswith('.cuh')]
    cudaObjsWithoutExtension = [each[:-4] for each in cudaObjs]

    for file in cudaObjsWithoutExtension:
        try:
            os.remove("./Build/" + file + ".dll")
            os.remove("./Includes/Libraries/" + file + ".lib")
        except:
            print("No copy of this file was built, file: " + file)
except:
    print("Couldn't load the directory path.")
