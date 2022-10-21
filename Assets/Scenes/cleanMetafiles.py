import os
import os.path
import sys

objs = []
objs += [".\Assets\Scenes\Meshes\\" + each for each in os.listdir(".\Assets\Scenes\Meshes") if each.endswith('.meta')]

files = [".\Assets\Scenes\\" + each for each in os.listdir(".\Assets\Scenes") if each.endswith('.coeus')]

result = []
for f in files:
    with open(f) as file:
        lines = file.readlines()
        lines = [line.rstrip().split(":") for line in lines]
        res = []
        for v in lines:
            if len(v) > 1 and v[1].endswith(".meta"):
                asd = v[1].split("\\")
                tmp = v[1].split("/")
                r = ""
                if len(asd) > len(tmp):
                    r = tmp[len(asd) - 1]
                else:
                    r = tmp[len(tmp) - 1]
                r = ".\Assets\Scenes\Meshes\\" + r
                res += [r]
    result += res

for obj in objs:
    exists = False
    for path in result:
        if obj == path:
            exists = True
    if not exists:
        os.remove(obj)

#for obj in objs:
#    if obj not in None:
#        os.remove()