import os

def findAllFiles():
    d = {}
    for path, subdirs, files in os.walk('.'):
        if "Includes" not in path:
            for name in files:
                if ".hpp" in name: # Getting all the .hpp files
                    with open(os.path.join(path, name)) as header:
                        if "public System::Component" in header.read():
                            d[name[:-4]] = list()
                            with open(os.path.join(path, name)) as serializableClass:
                                for line in serializableClass.readlines():
                                    if "serializable_" in line:
                                        d[name[:-4]] += [[line.split()[0], line.split()[1][:-1]]] 
    print(d)
            
findAllFiles()