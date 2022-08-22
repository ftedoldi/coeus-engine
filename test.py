import os

specialCharacters = { "//", "{", "}" }

def findAllFiles():
    d = {}
    for path, subdirs, files in os.walk('.'):
        if "Includes" not in path:
            for name in files:
                if ".hpp" in name: # Getting all the .hpp files
                    nmspace = ""
                    with open(os.path.join(path, name)) as header:
                        for line in header.readlines():
                            if "namespace" in line:
                                l = line.split()
                                l = [e for e in l if e not in specialCharacters]
                                nmspace = l[1]
                    with open(os.path.join(path, name)) as header:
                        if "public System::Component" in header.read():
                            d[name[:-4]] = nmspace
    # with open("./System/SerializableClass.hpp", 'w') as serializableClass:
    #         serializableClass.red
    print(d)
            
findAllFiles()