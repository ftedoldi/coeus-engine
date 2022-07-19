import os
import shutil

objs = []
objs += [each for each in os.listdir(".") if each.endswith('.obj')]

pdb = []
pdb += [each for each in os.listdir(".") if each.endswith('.pdb')]

try:
    for val in pdb:
        os.remove("./" + str(val))

finally:
    for element in objs:
        try:
            os.remove("./.objs/" + str(element))
        finally:
            shutil.move("./" + str(element), "./.objs/" + str(element))